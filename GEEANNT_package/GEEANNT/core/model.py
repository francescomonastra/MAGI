"""
Core model definitions for GEEANNT.

This module contains the main CVAE architecture used to model:
- energy (categorical)
- radial position
- angular position
- directional variable u_v
- angular direction phi_v
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .losses import (
    gaussian_nll,
    normalize_2d_pair,
    angular_loss_2d,
    smoothed_categorical_ce,
)
from .geometry import (
    xy_from_sr_phi,
    vxvy_from_uv_phi,
    u_r_from_sr,
)


class CVAE_CatEnergy_CatUV(keras.Model):
    """
    Conditional Variational Autoencoder with:
      - categorical energy head
      - Gaussian s_r head
      - categorical u_v head
      - 2D phi_r head
      - 2D phi_v head

    Continuous training input/output:
      y_cont = [s_r_s, cphi_r, sphi_r, cphi_v, sphi_v]

    Categorical targets:
      - energy_idx
      - uv_idx
    """

    def __init__(
        self,
        n_types,
        n_energy_bins,
        n_uv_bins,
        uv_bin_edges,
        latent_dim=8,
        hidden=(128, 128, 64),
        beta=0.1,
        type_weights=None,
        min_log_sigma=-6.0,
        max_log_sigma=1.5,
        lambda_sigma=1e-3,
        sigma_target=-2.0,
        lambda_phi=1e-2,
        lambda_phi_r=1e-2,
        w_energy=1.0,
        w_sr=1.0,
        w_uv=1.0,
        w_phi_r=1.0,
        w_phi_v=1.0,
        w_xy=0.0,
        w_vxy=0.05,
        w_ur=0.01,
        sphere_R=100.0,
        uv_label_smoothing=0.08,
        uv_neighbor_smoothing=True,
        uv_sampling_temperature=1.10,
        energy_sampling_temperature=1.00,
        sample_uv_uniform_inside_bin=True,
        phi_v_mse_weight=0.5,
        phi_v_ang_weight=1.2,
    ):
        super().__init__()

        self.n_types = n_types
        self.n_energy_bins = int(n_energy_bins)
        self.n_uv_bins = int(n_uv_bins)
        self.latent_dim = latent_dim
        self.beta = beta
        self.type_weights = type_weights

        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.lambda_sigma = lambda_sigma
        self.sigma_target = sigma_target
        self.lambda_phi = lambda_phi
        self.lambda_phi_r = lambda_phi_r

        self.w_energy = w_energy
        self.w_sr = w_sr
        self.w_uv = w_uv
        self.w_phi_r = w_phi_r
        self.w_phi_v = w_phi_v
        self.w_xy = w_xy
        self.w_vxy = w_vxy
        self.w_ur = w_ur

        self.sphere_R = sphere_R

        self.uv_label_smoothing = uv_label_smoothing
        self.uv_neighbor_smoothing = uv_neighbor_smoothing
        self.uv_sampling_temperature = uv_sampling_temperature
        self.energy_sampling_temperature = energy_sampling_temperature
        self.sample_uv_uniform_inside_bin = sample_uv_uniform_inside_bin

        self.phi_v_mse_weight = phi_v_mse_weight
        self.phi_v_ang_weight = phi_v_ang_weight

        uv_bin_edges = tf.convert_to_tensor(uv_bin_edges, dtype=tf.float32)
        self.uv_bin_edges = uv_bin_edges
        self.uv_bin_centers = 0.5 * (uv_bin_edges[:-1] + uv_bin_edges[1:])

        # ======================================================
        # Encoder
        # ======================================================
        enc_in = layers.Input(shape=(5 + n_energy_bins + n_uv_bins + n_types,))
        x = enc_in
        for h in hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.05)(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)

        self.encoder = keras.Model(enc_in, [z_mean, z_logvar], name="encoder")

        # ======================================================
        # Shared decoder backbone
        # ======================================================
        dec_in = layers.Input(shape=(latent_dim + n_types,))
        x = dec_in
        for h in hidden[::-1]:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.02)(x)

        self.decoder_backbone = keras.Model(dec_in, x, name="decoder_backbone")

        # ======================================================
        # Split decoder branches
        # ======================================================
        self.energy_branch = keras.Sequential([
            layers.Dense(64),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ], name="energy_branch")

        self.position_branch = keras.Sequential([
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ], name="position_branch")

        self.direction_branch = keras.Sequential([
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ], name="direction_branch")

        # ======================================================
        # Decoder heads
        # ======================================================
        self.energy_logits_head = layers.Dense(self.n_energy_bins, name="energy_logits")

        self.sr_head = keras.Sequential([
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="sr_head")

        self.sr_mu_head = layers.Dense(1, name="sr_mu")
        self.sr_logsigma_head = layers.Dense(1, name="sr_logsigma")

        self.uv_head = keras.Sequential([
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="uv_head")

        self.uv_logits_head = layers.Dense(self.n_uv_bins, name="uv_logits")

        self.phi_r_head = keras.Sequential([
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
            layers.Dense(2)
        ], name="phi_r_head")

        self.phi_v_head = keras.Sequential([
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
            layers.Dense(2)
        ], name="phi_v_head")

        # ======================================================
        # Metrics
        # ======================================================
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.rec_tracker = keras.metrics.Mean(name="rec")
        self.kl_tracker = keras.metrics.Mean(name="kl")
        self.nll_tracker = keras.metrics.Mean(name="nll")

        self.sigma_reg_tracker = keras.metrics.Mean(name="sigma_reg")
        self.phi_reg_tracker = keras.metrics.Mean(name="phi_reg")
        self.phi_r_reg_tracker = keras.metrics.Mean(name="phi_r_reg")

        self.energy_ce_tracker = keras.metrics.Mean(name="energy_ce")
        self.sr_nll_tracker = keras.metrics.Mean(name="sr_nll")
        self.uv_ce_tracker = keras.metrics.Mean(name="uv_ce")
        self.phi_r_mse_tracker = keras.metrics.Mean(name="phi_r_mse")
        self.phi_v_mse_tracker = keras.metrics.Mean(name="phi_v_mse")
        self.phi_v_ang_tracker = keras.metrics.Mean(name="phi_v_ang")
        self.phi_v_loss_tracker = keras.metrics.Mean(name="phi_v_loss")
        self.vxy_mse_tracker = keras.metrics.Mean(name="vxy_mse")
        self.xy_mse_tracker = keras.metrics.Mean(name="xy_mse")
        self.u_r_mse_tracker = keras.metrics.Mean(name="u_r_mse")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.rec_tracker,
            self.kl_tracker,
            self.nll_tracker,
            self.sigma_reg_tracker,
            self.phi_reg_tracker,
            self.phi_r_reg_tracker,
            self.energy_ce_tracker,
            self.sr_nll_tracker,
            self.uv_ce_tracker,
            self.phi_r_mse_tracker,
            self.phi_v_mse_tracker,
            self.phi_v_ang_tracker,
            self.phi_v_loss_tracker,
            self.vxy_mse_tracker,
            self.xy_mse_tracker,
            self.u_r_mse_tracker,
        ]

    # ==========================================================
    # Utilities
    # ==========================================================
    def sample_z(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def _sigma_regularizer(self, params):
        if self.lambda_sigma <= 0:
            return tf.constant(0.0, dtype=tf.float32)

        logsigmas = params["sr_logsigma"]
        return self.lambda_sigma * tf.reduce_mean(
            tf.square(tf.nn.relu(logsigmas - self.sigma_target))
        )

    def _phi_regularizers(self, params):
        phi_r_reg = tf.constant(0.0, tf.float32)
        if self.lambda_phi_r > 0:
            c = params["phi_r"][:, 0]
            s = params["phi_r"][:, 1]
            r2 = c * c + s * s
            phi_r_reg = self.lambda_phi_r * tf.reduce_mean(tf.square(r2 - 1.0))

        phi_v_reg = tf.constant(0.0, tf.float32)
        if self.lambda_phi > 0:
            c = params["phi_v"][:, 0]
            s = params["phi_v"][:, 1]
            r2 = c * c + s * s
            phi_v_reg = self.lambda_phi * tf.reduce_mean(tf.square(r2 - 1.0))

        return phi_v_reg, phi_r_reg

    def _expected_uv_from_logits(self, logits):
        probs = tf.nn.softmax(logits, axis=1)
        centers = tf.reshape(self.uv_bin_centers, [1, -1])
        uv_exp = tf.reduce_sum(probs * centers, axis=1)
        return uv_exp

    # ==========================================================
    # Decoder
    # ==========================================================
    def _decode_params(self, z, cond, training=False):
        shared = self.decoder_backbone(
            tf.concat([z, cond], axis=1),
            training=training
        )

        energy_feat = self.energy_branch(shared, training=training)
        pos_feat = self.position_branch(shared, training=training)
        dir_feat = self.direction_branch(shared, training=training)

        energy_logits = self.energy_logits_head(energy_feat)

        sr_feat = self.sr_head(pos_feat, training=training)
        sr_mu = self.sr_mu_head(sr_feat)
        sr_logsigma = tf.clip_by_value(
            self.sr_logsigma_head(sr_feat),
            self.min_log_sigma,
            self.max_log_sigma
        )

        phi_r = self.phi_r_head(pos_feat, training=training)

        uv_feat = self.uv_head(dir_feat, training=training)
        uv_logits = self.uv_logits_head(uv_feat)

        phi_v = self.phi_v_head(dir_feat, training=training)

        return {
            "energy_logits": energy_logits,
            "sr_mu": sr_mu,
            "sr_logsigma": sr_logsigma,
            "uv_logits": uv_logits,
            "phi_r": phi_r,
            "phi_v": phi_v,
        }

    def _reconstruction_terms(self, y_cont_true, E_idx_true, uv_idx_true, params):
        sr_true = y_cont_true[:, 0:1]
        phi_r_true = y_cont_true[:, 1:3]
        phi_v_true = y_cont_true[:, 3:5]

        energy_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=E_idx_true,
            logits=params["energy_logits"]
        )

        sr_nll = tf.squeeze(
            gaussian_nll(sr_true, params["sr_mu"], params["sr_logsigma"]),
            axis=1
        )

        uv_ce = smoothed_categorical_ce(
            labels=uv_idx_true,
            logits=params["uv_logits"],
            n_classes=self.n_uv_bins,
            smoothing=self.uv_label_smoothing,
            local_neighbors=self.uv_neighbor_smoothing
        )

        phi_r_mse = tf.reduce_sum(tf.square(phi_r_true - params["phi_r"]), axis=1)

        phi_v_pred_unit = normalize_2d_pair(params["phi_v"])
        phi_v_mse = tf.reduce_sum(tf.square(phi_v_true - phi_v_pred_unit), axis=1)
        phi_v_ang = angular_loss_2d(phi_v_true, params["phi_v"])

        phi_v_loss = (
            self.phi_v_mse_weight * phi_v_mse +
            self.phi_v_ang_weight * phi_v_ang
        )

        x_true, y_true_xy = xy_from_sr_phi(sr_true, phi_r_true, self.sphere_R)
        x_pred, y_pred_xy = xy_from_sr_phi(params["sr_mu"], params["phi_r"], self.sphere_R)
        xy_mse = tf.square(x_true - x_pred) + tf.square(y_true_xy - y_pred_xy)

        u_r_true = u_r_from_sr(sr_true)
        u_r_pred = u_r_from_sr(params["sr_mu"])
        u_r_mse = tf.square(u_r_true - u_r_pred)

        uv_true = tf.gather(self.uv_bin_centers, uv_idx_true)
        uv_pred = self._expected_uv_from_logits(params["uv_logits"])

        vx_true, vy_true = vxvy_from_uv_phi(uv_true, phi_v_true)
        vx_pred, vy_pred = vxvy_from_uv_phi(uv_pred, params["phi_v"])
        vxy_mse = tf.square(vx_true - vx_pred) + tf.square(vy_true - vy_pred)

        rec_per = (
            self.w_energy * energy_ce +
            self.w_sr     * sr_nll +
            self.w_uv     * uv_ce +
            self.w_phi_r  * phi_r_mse +
            self.w_phi_v  * phi_v_loss +
            self.w_xy     * xy_mse +
            self.w_ur     * u_r_mse +
            self.w_vxy    * vxy_mse
        )

        pieces = {
            "energy_ce": energy_ce,
            "sr_nll": sr_nll,
            "uv_ce": uv_ce,
            "phi_r_mse": phi_r_mse,
            "phi_v_mse": phi_v_mse,
            "phi_v_ang": phi_v_ang,
            "phi_v_loss": phi_v_loss,
            "xy_mse": xy_mse,
            "u_r_mse": u_r_mse,
            "vxy_mse": vxy_mse,
        }
        return rec_per, pieces

    # ==========================================================
    # Keras train/test
    # ==========================================================
    def train_step(self, data):
        (y_cont, E_idx_true, uv_idx_true, cond), _ = data

        E_onehot = tf.one_hot(E_idx_true, depth=self.n_energy_bins, dtype=tf.float32)
        uv_onehot = tf.one_hot(uv_idx_true, depth=self.n_uv_bins, dtype=tf.float32)

        x_in = tf.concat([y_cont, E_onehot, uv_onehot, cond], axis=1)

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(x_in, training=True)
            z = self.sample_z(z_mean, z_logvar)

            params = self._decode_params(z, cond, training=True)
            rec_per, pieces = self._reconstruction_terms(y_cont, E_idx_true, uv_idx_true, params)

            if self.type_weights is not None:
                t_idx = tf.argmax(cond, axis=1, output_type=tf.int32)
                w = tf.gather(self.type_weights, t_idx)
                rec_per_weighted = rec_per * w
            else:
                rec_per_weighted = rec_per

            rec = tf.reduce_mean(rec_per_weighted)
            nll = tf.reduce_mean(rec_per)

            kl_per = -0.5 * tf.reduce_sum(
                1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar),
                axis=1
            )
            kl = tf.reduce_mean(kl_per)

            sigma_reg = self._sigma_regularizer(params)
            phi_reg, phi_r_reg = self._phi_regularizers(params)

            loss = rec + self.beta * kl + sigma_reg + phi_reg + phi_r_reg

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        self.nll_tracker.update_state(nll)

        self.sigma_reg_tracker.update_state(sigma_reg)
        self.phi_reg_tracker.update_state(phi_reg)
        self.phi_r_reg_tracker.update_state(phi_r_reg)

        self.energy_ce_tracker.update_state(tf.reduce_mean(pieces["energy_ce"]))
        self.sr_nll_tracker.update_state(tf.reduce_mean(pieces["sr_nll"]))
        self.uv_ce_tracker.update_state(tf.reduce_mean(pieces["uv_ce"]))
        self.phi_r_mse_tracker.update_state(tf.reduce_mean(pieces["phi_r_mse"]))
        self.phi_v_mse_tracker.update_state(tf.reduce_mean(pieces["phi_v_mse"]))
        self.phi_v_ang_tracker.update_state(tf.reduce_mean(pieces["phi_v_ang"]))
        self.phi_v_loss_tracker.update_state(tf.reduce_mean(pieces["phi_v_loss"]))
        self.vxy_mse_tracker.update_state(tf.reduce_mean(pieces["vxy_mse"]))
        self.xy_mse_tracker.update_state(tf.reduce_mean(pieces["xy_mse"]))
        self.u_r_mse_tracker.update_state(tf.reduce_mean(pieces["u_r_mse"]))

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (y_cont, E_idx_true, uv_idx_true, cond), _ = data

        E_onehot = tf.one_hot(E_idx_true, depth=self.n_energy_bins, dtype=tf.float32)
        uv_onehot = tf.one_hot(uv_idx_true, depth=self.n_uv_bins, dtype=tf.float32)

        x_in = tf.concat([y_cont, E_onehot, uv_onehot, cond], axis=1)

        z_mean, z_logvar = self.encoder(x_in, training=False)
        z = self.sample_z(z_mean, z_logvar)

        params = self._decode_params(z, cond, training=False)
        rec_per, pieces = self._reconstruction_terms(y_cont, E_idx_true, uv_idx_true, params)

        if self.type_weights is not None:
            t_idx = tf.argmax(cond, axis=1, output_type=tf.int32)
            w = tf.gather(self.type_weights, t_idx)
            rec_per_weighted = rec_per * w
        else:
            rec_per_weighted = rec_per

        rec = tf.reduce_mean(rec_per_weighted)
        nll = tf.reduce_mean(rec_per)

        kl_per = -0.5 * tf.reduce_sum(
            1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar),
            axis=1
        )
        kl = tf.reduce_mean(kl_per)

        sigma_reg = self._sigma_regularizer(params)
        phi_reg, phi_r_reg = self._phi_regularizers(params)

        loss = rec + self.beta * kl + sigma_reg + phi_reg + phi_r_reg

        self.loss_tracker.update_state(loss)
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        self.nll_tracker.update_state(nll)

        self.sigma_reg_tracker.update_state(sigma_reg)
        self.phi_reg_tracker.update_state(phi_reg)
        self.phi_r_reg_tracker.update_state(phi_r_reg)

        self.energy_ce_tracker.update_state(tf.reduce_mean(pieces["energy_ce"]))
        self.sr_nll_tracker.update_state(tf.reduce_mean(pieces["sr_nll"]))
        self.uv_ce_tracker.update_state(tf.reduce_mean(pieces["uv_ce"]))
        self.phi_r_mse_tracker.update_state(tf.reduce_mean(pieces["phi_r_mse"]))
        self.phi_v_mse_tracker.update_state(tf.reduce_mean(pieces["phi_v_mse"]))
        self.phi_v_ang_tracker.update_state(tf.reduce_mean(pieces["phi_v_ang"]))
        self.phi_v_loss_tracker.update_state(tf.reduce_mean(pieces["phi_v_loss"]))
        self.vxy_mse_tracker.update_state(tf.reduce_mean(pieces["vxy_mse"]))
        self.xy_mse_tracker.update_state(tf.reduce_mean(pieces["xy_mse"]))
        self.u_r_mse_tracker.update_state(tf.reduce_mean(pieces["u_r_mse"]))

        return {m.name: m.result() for m in self.metrics}

    # ==========================================================
    # Sampling / generation
    # ==========================================================
    @tf.function(reduce_retracing=True)
    def decode(self, z, cond):
        return self._decode_params(z, cond, training=False)

    def _sample_energy_from_logits(self, logits):
        logits = logits / self.energy_sampling_temperature
        idx = tf.random.categorical(logits, num_samples=1)
        idx = tf.cast(tf.squeeze(idx, axis=1), tf.int32)
        return idx

    def _sample_uv_from_logits(self, logits):
        logits = logits / self.uv_sampling_temperature
        idx = tf.random.categorical(logits, num_samples=1)
        idx = tf.cast(tf.squeeze(idx, axis=1), tf.int32)
        return idx

    def _sample_uv_value_from_idx(self, uv_idx):
        left = tf.gather(self.uv_bin_edges[:-1], uv_idx)
        right = tf.gather(self.uv_bin_edges[1:], uv_idx)

        if self.sample_uv_uniform_inside_bin:
            u = tf.random.uniform(tf.shape(left), 0.0, 1.0, dtype=tf.float32)
            uv_value = left + u * (right - left)
        else:
            uv_value = 0.5 * (left + right)

        return tf.expand_dims(uv_value, axis=1)

    def generate(self, cond, n_samples):
        z = tf.random.normal((n_samples, self.latent_dim))
        params = self.decode(z, cond)

        energy_idx = self._sample_energy_from_logits(params["energy_logits"])

        sr_eps = tf.random.normal(tf.shape(params["sr_mu"]))
        sr_s = params["sr_mu"] + tf.exp(params["sr_logsigma"]) * sr_eps

        uv_idx = self._sample_uv_from_logits(params["uv_logits"])
        uv_value = self._sample_uv_value_from_idx(uv_idx)

        phi_r = params["phi_r"]
        phi_v = params["phi_v"]

        y_cont = tf.concat([sr_s, phi_r, phi_v], axis=1)

        return {
            "energy_idx": energy_idx,
            "uv_idx": uv_idx,
            "uv_value": uv_value,
            "y_cont": y_cont,
            "params": params,
        }