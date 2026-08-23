"""
Conditional learnable prior p(z | cond) for the v0.8 mixture energy head
(CVAE_MixEnergy_ContPhi_TaskAdaptive).

Why this exists
---------------
The physically-correct default for that model is energy_flow_condition="z_cond":
the energy flow and gate condition on the per-event latent z, so that energy stays
correlated with geometry (a given energy - especially a material-dependent
fluorescence line - is correlated with the volume/geometry it came from, all
mediated by z). But a *high-capacity* z-conditioned flow re-exposes the classic
aggregated-posterior / prior mismatch: it leaks per-event energy through z to drive
the training NLL down, and generation (which samples z ~ N(0, I)) then feeds the
decoder z's from regions it never trained on, so the generated spectrum degrades the
longer we train (see docs/v0.8_learnable_prior_plan.md sections 2-4 and
docs/v0.8_fixing_plan.md section 8).

The fix is to stop *assuming* the prior is N(0, I) and instead *learn* the actual
distribution of latents the encoder produces (the aggregated posterior), conditioned
on the particle type cond, then sample from that at generation. That learned
distribution is this module: a conditional coupling-flow prior. It keeps the z-
coupling intact (the decoder still reads z) while making the sampled z realistic, so
generation is fixed for every head at once.

Why coupling (not IAF / MAF)
----------------------------
The prior is used in *both* directions: sampled at generation and log-prob-scored in
the KL at training. Coupling flows are cheap in both directions, unlike autoregressive
flows which are cheap in only one. We use affine (RealNVP-style) coupling layers over
the latent vector z: each layer passes half the dimensions through unchanged and uses
them (together with cond) to compute an affine shift+scale for the other half; the
mask alternates between layers so every dimension gets transformed. The log-Jacobian
is just the sum of the log-scales - trivial and stable. (Escalation to spline coupling
is possible if affine is not expressive enough - see the fixing plan - but affine is
the first attempt.)

Base distribution is a standard normal in latent_dim dimensions. The final Dense of
every conditioner is zero-initialized, so at initialization every layer is the
identity and the prior is exactly N(0, I) - i.e. byte-compatible with the gaussian-
prior baseline at step 0, letting the prior break away from N(0, I) only as training
demands.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ConditionalCouplingPrior(keras.layers.Layer):
    """Conditional affine coupling-flow prior p(z | cond) over an 8-D latent.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent z (8 for MAGI).
    cond_dim : int
        Width of the conditioning vector cond (n_types, the particle-type
        one-hot).
    n_layers : int
        Number of affine coupling layers. Masks alternate each layer, so an
        even count gives every dimension the same number of transforms.
    hidden : tuple of int
        Hidden layer widths of each coupling conditioner MLP.
    log_scale_clamp : float
        Bound on the per-dimension log-scale: log_scale = clamp * tanh(raw),
        keeping exp(log_scale) in [exp(-clamp), exp(clamp)] for numerical
        stability (as in NormalFlow_Fioretti/increaseSPO_L2_NF.py).
    """

    def __init__(
        self,
        latent_dim=8,
        cond_dim=1,
        n_layers=6,
        hidden=(64, 64),
        log_scale_clamp=3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.latent_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)
        self.n_layers = int(n_layers)
        self.hidden = tuple(hidden)
        self.log_scale_clamp = float(log_scale_clamp)

        # Alternating binary masks. mask == 1 => dimension is passed through
        # unchanged and feeds the conditioner; mask == 0 => dimension is the
        # one being affine-transformed this layer.
        base_mask = tf.constant(
            [1.0 if (i % 2 == 0) else 0.0 for i in range(self.latent_dim)],
            dtype=tf.float32,
        )
        self._masks = []
        for i in range(self.n_layers):
            self._masks.append(base_mask if (i % 2 == 0) else 1.0 - base_mask)

        # One conditioner MLP per layer: [masked z, cond] -> (shift, log_scale),
        # each of width latent_dim (concatenated -> 2 * latent_dim outputs).
        # Zero-init the final layer so each layer starts as the identity and the
        # prior starts at exactly N(0, I).
        self._conditioners = []
        for i in range(self.n_layers):
            mlp = keras.Sequential(name=f"{self.name}_coupling_{i}")
            mlp.add(layers.Input(shape=(self.latent_dim + self.cond_dim,)))
            for h in self.hidden:
                mlp.add(layers.Dense(h))
                mlp.add(layers.LeakyReLU(0.05))
            mlp.add(
                layers.Dense(
                    2 * self.latent_dim,
                    kernel_initializer="zeros",
                    bias_initializer="zeros",
                )
            )
            self._conditioners.append(mlp)

    def _shift_log_scale(self, layer_idx, v, cond):
        """Affine params (shift, log_scale) for coupling layer `layer_idx`.

        Both are zeroed on the masked (pass-through) dimensions so only the
        transformed half is affected.
        """
        mask = self._masks[layer_idx]
        v_masked = v * mask
        net_in = tf.concat([v_masked, cond], axis=1)
        raw = self._conditioners[layer_idx](net_in)
        shift, raw_scale = tf.split(raw, 2, axis=1)
        log_scale = self.log_scale_clamp * tf.tanh(raw_scale)
        keep = 1.0 - mask
        return shift * keep, log_scale * keep

    def _forward(self, u, cond):
        """Map base u -> z through all layers (used for sampling)."""
        z = u
        for i in range(self.n_layers):
            shift, log_scale = self._shift_log_scale(i, z, cond)
            z = z * tf.exp(log_scale) + shift
        return z

    def _inverse(self, z, cond):
        """Map z -> base u through all layers in reverse; return (u, sum_log_det).

        sum_log_det is the log|det dz/du| accumulated forward-style
        (sum of log_scale); the change-of-variables term for log p(z) is its
        negative.
        """
        u = z
        sum_log_det = tf.zeros((tf.shape(z)[0],), dtype=z.dtype)
        for i in reversed(range(self.n_layers)):
            # The masked (pass-through) dims of u equal those of z, so the
            # conditioner sees the same input as in the forward pass.
            shift, log_scale = self._shift_log_scale(i, u, cond)
            u = (u - shift) * tf.exp(-log_scale)
            sum_log_det += tf.reduce_sum(log_scale, axis=1)
        return u, sum_log_det

    def log_prob(self, z, cond):
        """Fully normalized log density log p(z | cond), shape (batch,)."""
        u, sum_log_det = self._inverse(z, cond)
        # Standard-normal base, log density summed over latent dims.
        log_base = -0.5 * tf.reduce_sum(
            tf.square(u) + tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=u.dtype)),
            axis=1,
        )
        # log p_Z(z) = log p_U(u) - log|det dz/du|.
        return log_base - sum_log_det

    def sample(self, cond):
        """Draw one z per row of cond, shape (batch, latent_dim)."""
        n = tf.shape(cond)[0]
        u = tf.random.normal((n, self.latent_dim))
        return self._forward(u, cond)

    def get_config(self):
        """Serialize the prior's construction arguments for save/load.

        Keeping this complete matters: the prior is rebuilt from this dict at
        generation time, and a missing key silently yields a different
        architecture from the one that was trained.
        """
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "cond_dim": self.cond_dim,
                "n_layers": self.n_layers,
                "hidden": list(self.hidden),
                "log_scale_clamp": self.log_scale_clamp,
            }
        )
        return config
