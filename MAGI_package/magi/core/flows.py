"""
Conditional normalizing-flow density for the continuum term of the v0.8
mixture energy head (CVAE_MixEnergy_ContPhi_TaskAdaptive).

A single Gaussian - or even a K-component Gaussian sub-mixture - continuum
cannot represent both a sharp, low-entropy peak (CryoSphere-Small) and a
broad multi-modal continuum (CryoSphere-CR) without a regularizer trade-off
that fails one to satisfy the other (see docs/v0.8_fixing_plan.md sections
2b and 7). This module supplies the escalation recommended there (section
3(b)) and in docs/v0.8_IAF_integration_plan.md section 7: replace the
parametric continuum density with a conditional normalizing flow, leaving
the fixed discrete line components untouched.

The target here is a *scalar* (transformed energy y = T(E)), so a
RealNVP-style coupling flow does not apply - coupling splits a vector into
halves that condition each other, which a 1-D variable has none of. The
1-D analog with the same "cheap in both directions" property is a
conditional monotonic rational-quadratic spline (Durkan et al. 2019): the
spline knot parameters are produced by a small conditioner network from a
conditioning vector `feat` (never from y), so the per-sample map y <-> u is
an expressive monotonic bijection whose log_prob and sample are each a
single pass.

The caller decides what `feat` is. In CVAE_MixEnergy_ContPhi_TaskAdaptive
the continuum flow conditions on `cond` (the particle-type one-hot), NOT on
the per-event latent z: the continuum shape is a property of the source, and
conditioning on z lets the flexible flow leak per-event energy through z,
which then diverges at generation (z ~ N(0, I)) - see that model's flow-head
construction comment and docs/v0.8_IAF_integration_plan.md sections 2 and 4.

Working space and tails
------------------------
The spline warps a fixed interval [-B, B] in a *standardized* space
y_std = (y - y_mean) / y_scale; outside [-B, B] tfp's RationalQuadraticSpline
is the identity (verified), so an unbounded N(0, 1) base never lands in an
undefined region and the flow is a valid bijection on all of R with
standard-normal tails. The caller must pass y_mean / y_scale measured from
the training energy_y (e.g. its mean / std) so the data bulk sits well
inside [-B, B]; B defaults to 5 (~5 sigma of coverage).

log_prob returns a *fully normalized* log density in y-space (the base
Normal's log(2*pi) constant included), so it can be mixed with the line
components' log densities in one log-sum-exp only if those line densities
are also fully normalized - see core.losses.flow_line_mixture_nll /
gaussian_logpdf.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers

tfb = tfp.bijectors
tfd = tfp.distributions


def _piecewise_linear(x, xk, yk):
    """Monotone piecewise-linear map x -> y through knots (xk, yk), both 1-D
    and strictly increasing. Returns (y, slope) where slope is the local
    segment slope dy/dx at each x. Points outside [xk[0], xk[-1]] are
    extrapolated with the nearest end-segment slope (still monotone)."""
    x = tf.reshape(x, (-1,))
    m = tf.shape(xk)[0]
    idx = tf.searchsorted(xk, x, side="right") - 1
    idx = tf.clip_by_value(idx, 0, m - 2)
    x0 = tf.gather(xk, idx)
    x1 = tf.gather(xk, idx + 1)
    y0 = tf.gather(yk, idx)
    y1 = tf.gather(yk, idx + 1)
    slope = (y1 - y0) / (x1 - x0)
    y = y0 + (x - x0) * slope
    return y, slope


class ConditionalRQSFlow(keras.layers.Layer):
    """Conditional 1-D rational-quadratic-spline flow p(y | feat).

    Parameters
    ----------
    feat_dim : int
        Width of the conditioning feature vector (the decoder's
        energy_cont_feat, shape (batch, feat_dim)).
    y_mean, y_scale : float
        Standardization of y into the spline's working space
        y_std = (y - y_mean) / y_scale. Should be measured from the
        training energy_y (mean / std) so the bulk sits inside [-B, B].
    n_bins : int
        Number of spline bins per transform.
    n_transforms : int
        Number of stacked spline transforms (with a sign flip between
        successive transforms for extra flexibility).
    interval_half_width : float
        Half-width B of the spline domain [-B, B] in standardized space.
    conditioner_hidden : tuple of int
        Hidden layer widths of the conditioner MLP.
    min_bin : float
        Floor on each (normalized) bin width/height, so no bin can collapse
        to zero. Must satisfy min_bin * n_bins < 1.
    min_slope : float
        Floor on the knot slopes (kept strictly positive for monotonicity).
    """

    def __init__(
        self,
        feat_dim,
        y_mean,
        y_scale,
        n_bins=8,
        n_transforms=2,
        interval_half_width=5.0,
        conditioner_hidden=(64,),
        min_bin=1e-3,
        min_slope=1e-3,
        warp_mode="affine",
        warp_y_knots=None,
        warp_z_knots=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.feat_dim = int(feat_dim)
        self.y_mean = float(y_mean)
        self.y_scale = float(y_scale)
        if self.y_scale <= 0.0:
            raise ValueError(f"y_scale must be positive, got {self.y_scale}.")

        # Standardization of y into the spline's working space.
        #   "affine": w = (y - y_mean) / y_scale  (constant Jacobian; default,
        #             byte-for-byte the original behavior).
        #   "cdf":    w = monotone piecewise-linear map fit so the training
        #             energy_y marginal becomes ~N(0,1) (knots = data quantiles
        #             -> standard-normal quantiles). This makes the spline knots
        #             density-proportional, so a broad multi-scale spectrum (CR:
        #             sharp low-E Compton edge under a huge muon tail) and a
        #             narrow bulk with a far low-E tail (Small) both land inside
        #             [-B, B] with resolution where the events are. See
        #             docs/v0.8_v072_comparison.md section 6(b).
        self.warp_mode = str(warp_mode)
        if self.warp_mode not in ("affine", "cdf"):
            raise ValueError(
                f"warp_mode must be 'affine' or 'cdf', got {self.warp_mode!r}."
            )
        if self.warp_mode == "cdf":
            if warp_y_knots is None or warp_z_knots is None:
                raise ValueError(
                    "warp_mode='cdf' requires warp_y_knots and warp_z_knots "
                    "(fit with magi.fit_cdf_warp_knots)."
                )
            yk = np.asarray(warp_y_knots, dtype=np.float64).reshape(-1)
            zk = np.asarray(warp_z_knots, dtype=np.float64).reshape(-1)
            if yk.shape != zk.shape or yk.shape[0] < 2:
                raise ValueError(
                    "warp_y_knots and warp_z_knots must be 1-D of equal length >= 2."
                )
            if np.any(np.diff(yk) <= 0.0) or np.any(np.diff(zk) <= 0.0):
                raise ValueError("warp knots must be strictly increasing.")
            self.warp_y_knots_np = yk
            self.warp_z_knots_np = zk
            self._yk = tf.constant(yk, dtype=tf.float32)
            self._zk = tf.constant(zk, dtype=tf.float32)
        else:
            self.warp_y_knots_np = None
            self.warp_z_knots_np = None

        self.n_bins = int(n_bins)
        self.n_transforms = int(n_transforms)
        self.B = float(interval_half_width)
        self.min_bin = float(min_bin)
        self.min_slope = float(min_slope)
        if self.min_bin * self.n_bins >= 1.0:
            raise ValueError(
                "min_bin * n_bins must be < 1 "
                f"(got min_bin={self.min_bin}, n_bins={self.n_bins})."
            )

        # Per transform: n_bins widths + n_bins heights + (n_bins - 1) slopes.
        self._params_per_transform = 3 * self.n_bins - 1
        self._n_params = self.n_transforms * self._params_per_transform

        conditioner = keras.Sequential(name=f"{self.name}_conditioner")
        conditioner.add(layers.Input(shape=(self.feat_dim,)))
        for h in conditioner_hidden:
            conditioner.add(layers.Dense(h))
            conditioner.add(layers.LeakyReLU(0.05))
        # Zero-init the final layer so every transform starts near uniform
        # (near-identity), letting the flow break symmetry during training
        # rather than from an arbitrary random warp.
        conditioner.add(
            layers.Dense(
                self._n_params,
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )
        )
        self.conditioner = conditioner

        self.base = tfd.Normal(loc=0.0, scale=1.0)

    def _build_bijector(self, feat):
        """Per-sample chained RQS bijector (forward maps base u -> y_std)."""
        raw = self.conditioner(feat)
        raw = tf.reshape(raw, (-1, self.n_transforms, self._params_per_transform))

        span = 2.0 * self.B
        floor_scale = 1.0 - self.min_bin * self.n_bins

        bijectors = []
        for t in range(self.n_transforms):
            p = raw[:, t, :]
            raw_w = p[:, : self.n_bins]
            raw_h = p[:, self.n_bins : 2 * self.n_bins]
            raw_s = p[:, 2 * self.n_bins :]

            widths = (self.min_bin + floor_scale * tf.nn.softmax(raw_w, axis=-1)) * span
            heights = (self.min_bin + floor_scale * tf.nn.softmax(raw_h, axis=-1)) * span
            slopes = self.min_slope + tf.nn.softplus(raw_s)

            bijectors.append(
                tfb.RationalQuadraticSpline(
                    bin_widths=widths,
                    bin_heights=heights,
                    knot_slopes=slopes,
                    range_min=-self.B,
                )
            )
            if t < self.n_transforms - 1:
                # Sign flip keeps [-B, B] mapped to itself while letting the
                # next spline act on a reversed coordinate.
                bijectors.append(tfb.Scale(-1.0))

        return tfb.Chain(bijectors)

    def _warp_forward(self, y):
        """Map y -> standardized working coordinate w, returning
        (w, log|dw/dy|), both shape (batch,)."""
        y = tf.reshape(y, (-1,))
        if self.warp_mode == "affine":
            w = (y - self.y_mean) / self.y_scale
            log_dwdy = -tf.math.log(tf.constant(self.y_scale, dtype=w.dtype))
            return w, log_dwdy
        w, slope = _piecewise_linear(y, self._yk, self._zk)
        return w, tf.math.log(slope)

    def _warp_inverse(self, w):
        """Map standardized working coordinate w -> y, shape (batch,)."""
        if self.warp_mode == "affine":
            return tf.reshape(w, (-1,)) * self.y_scale + self.y_mean
        y, _ = _piecewise_linear(w, self._zk, self._yk)
        return y

    def log_prob(self, y, feat):
        """Fully normalized log density log p(y | feat), shape (batch,)."""
        w, log_dwdy = self._warp_forward(y)
        bijector = self._build_bijector(feat)

        u = bijector.inverse(w)
        log_prob_std = self.base.log_prob(u) + bijector.inverse_log_det_jacobian(
            w, event_ndims=0
        )
        # Change of variables for the standardization: p_Y(y) = p_W(w) |dw/dy|.
        # affine: log_dwdy = -log(y_scale) (constant) -> identical to before.
        return log_prob_std + log_dwdy

    def sample(self, feat):
        """Draw one y per row of feat, shape (batch,)."""
        n = tf.shape(feat)[0]
        u = self.base.sample(n)
        bijector = self._build_bijector(feat)
        w = bijector.forward(u)
        return self._warp_inverse(w)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "feat_dim": self.feat_dim,
                "y_mean": self.y_mean,
                "y_scale": self.y_scale,
                "n_bins": self.n_bins,
                "n_transforms": self.n_transforms,
                "interval_half_width": self.B,
                "min_bin": self.min_bin,
                "min_slope": self.min_slope,
                "warp_mode": self.warp_mode,
                "warp_y_knots": (
                    None if self.warp_y_knots_np is None
                    else self.warp_y_knots_np.tolist()
                ),
                "warp_z_knots": (
                    None if self.warp_z_knots_np is None
                    else self.warp_z_knots_np.tolist()
                ),
            }
        )
        return config
