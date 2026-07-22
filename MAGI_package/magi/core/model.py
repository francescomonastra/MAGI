"""
Core model definitions for MAGI.

This module contains the main CVAE architecture used to model:
- energy (categorical)
- radial position
- angular position
- directional variable u_v
- angular direction phi_v
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .losses import (
    gaussian_nll,
    gaussian_mixture_nll,
    flow_line_mixture_nll,
    normalize_2d_pair,
    angular_loss_2d,
    smoothed_categorical_ce,
)
from .flows import ConditionalRQSFlow
from .geometry import (
    xy_from_sr_phi,
    vxvy_from_uv_phi,
    u_r_from_sr,
    xy_from_ur_phi,
    xyz_from_ur_phi,
    vxyz_from_uv_phi,
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
    


## =======================================================
## Task-adaptive CVAE variant
## =======================================================

"""
Task-adaptive CVAE model definitions for MAGI.

This variant is designed to:
- keep easy tasks on lighter branches
- reserve deeper decoder capacity for harder tasks
- support dynamic loss-weight adaptation during training
"""


class CVAE_CatEnergy_CatUV_TaskAdaptive(keras.Model):
    """
    Conditional Variational Autoencoder with task-adaptive decoder design.

    Main idea:
    - a light shared decoder stem
    - an energy branch attached to the light stem
    - a deeper decoder trunk reserved for harder tasks
    - mutable task weights for adaptive training

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

        # Gaussian head control
        min_log_sigma=-6.0,
        max_log_sigma=1.5,

        # Regularization
        lambda_sigma=1e-3,
        sigma_target=-2.0,
        lambda_phi=1e-2,
        lambda_phi_r=1e-2,

        # Reconstruction weights
        w_energy=1.0,
        w_sr=1.0,
        w_uv=1.0,
        w_phi_r=1.0,
        w_phi_v=1.0,
        w_xy=0.0,
        w_vxy=0.05,
        w_ur=0.01,

        # Geometry
        sphere_R=100.0,

        # Controls for u_v categorical modeling
        uv_label_smoothing=0.08,
        uv_neighbor_smoothing=True,
        uv_sampling_temperature=1.10,
        energy_sampling_temperature=1.00,
        sample_uv_uniform_inside_bin=True,

        # Relative weights inside phi_v loss
        phi_v_mse_weight=0.5,
        phi_v_ang_weight=1.2,

        # Task-adaptive architecture controls
        stem_width=64,
        deep_decoder_hidden=(128, 128, 64),
        energy_branch_hidden=(48, 48),
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

        self.w_energy = float(w_energy)
        self.w_sr = float(w_sr)
        self.w_uv = float(w_uv)
        self.w_phi_r = float(w_phi_r)
        self.w_phi_v = float(w_phi_v)
        self.w_xy = float(w_xy)
        self.w_vxy = float(w_vxy)
        self.w_ur = float(w_ur)

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
        # Mutable task weights dictionary
        # ======================================================
        self.task_weights = {
            "energy": self.w_energy,
            "sr": self.w_sr,
            "uv": self.w_uv,
            "phi_r": self.w_phi_r,
            "phi_v": self.w_phi_v,
            "xy": self.w_xy,
            "vxy": self.w_vxy,
            "ur": self.w_ur,
        }

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
        # Light shared decoder stem
        # ======================================================
        dec_in = layers.Input(shape=(latent_dim + n_types,))
        stem = layers.Dense(stem_width)(dec_in)
        stem = layers.LeakyReLU(0.03)(stem)
        self.decoder_stem = keras.Model(dec_in, stem, name="decoder_stem")

        # ======================================================
        # Deep decoder trunk (reserved for harder tasks)
        # ======================================================
        trunk_in = layers.Input(shape=(stem_width,))
        x = trunk_in
        for h in deep_decoder_hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.03)(x)
        self.decoder_deep_trunk = keras.Model(trunk_in, x, name="decoder_deep_trunk")

        # ======================================================
        # Task branches
        # ======================================================

        # Easy / early-converging task: energy
        energy_in = layers.Input(shape=(stem_width,))
        x = energy_in
        for h in energy_branch_hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.03)(x)
        self.energy_branch = keras.Model(energy_in, x, name="energy_branch")

        # Harder tasks: position and direction
        self.position_branch = keras.Sequential([
            layers.Input(shape=(deep_decoder_hidden[-1],)),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ], name="position_branch")

        self.direction_branch = keras.Sequential([
            layers.Input(shape=(deep_decoder_hidden[-1],)),
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
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="sr_head")
        self.sr_mu_head = layers.Dense(1, name="sr_mu")
        self.sr_logsigma_head = layers.Dense(1, name="sr_logsigma")

        self.uv_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="uv_head")
        self.uv_logits_head = layers.Dense(self.n_uv_bins, name="uv_logits")

        self.phi_r_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
            layers.Dense(2),
        ], name="phi_r_head")

        self.phi_v_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
            layers.Dense(2),
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
    # Task-weight API
    # ==========================================================
    def get_task_weight(self, task_name):
        return float(self.task_weights[task_name])

    def set_task_weight(self, task_name, value):
        value = float(value)
        self.task_weights[task_name] = value

        if task_name == "energy":
            self.w_energy = value
        elif task_name == "sr":
            self.w_sr = value
        elif task_name == "uv":
            self.w_uv = value
        elif task_name == "phi_r":
            self.w_phi_r = value
        elif task_name == "phi_v":
            self.w_phi_v = value
        elif task_name == "xy":
            self.w_xy = value
        elif task_name == "vxy":
            self.w_vxy = value
        elif task_name == "ur":
            self.w_ur = value
        else:
            raise ValueError(f"Unknown task name: {task_name}")

    def decay_task_weight(self, task_name, factor=0.5, min_value=0.0):
        old = self.get_task_weight(task_name)
        new = max(min_value, old * float(factor))
        self.set_task_weight(task_name, new)
        return old, new

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
        base = tf.concat([z, cond], axis=1)

        stem = self.decoder_stem(base, training=training)
        deep = self.decoder_deep_trunk(stem, training=training)

        energy_feat = self.energy_branch(stem, training=training)
        pos_feat = self.position_branch(deep, training=training)
        dir_feat = self.direction_branch(deep, training=training)

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
            self.task_weights["energy"] * energy_ce +
            self.task_weights["sr"]     * sr_nll +
            self.task_weights["uv"]     * uv_ce +
            self.task_weights["phi_r"]  * phi_r_mse +
            self.task_weights["phi_v"]  * phi_v_loss +
            self.task_weights["xy"]     * xy_mse +
            self.task_weights["ur"]     * u_r_mse +
            self.task_weights["vxy"]    * vxy_mse
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

        # ----------------------------------------------------------
        # Safe gradient application:
        # - drop None gradients
        # - deduplicate variables in case nested submodels expose
        #   the same variable more than once
        # ----------------------------------------------------------
        grads_and_vars = []
        seen = set()

        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue

            vid = id(v)
            if vid in seen:
                continue

            seen.add(vid)
            grads_and_vars.append((g, v))

        self.optimizer.apply_gradients(grads_and_vars)

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
    



## =======================================================
## Task-adaptive CVAE variant with Quantile Transformation
## =======================================================

"""
Task-adaptive CVAE model definitions for MAGI.

This variant is designed to:
- keep easy tasks on lighter branches
- reserve deeper decoder capacity for harder tasks
- support dynamic loss-weight adaptation during training
- use as input u_r_q and u_v_q quantiles transformed to reduce dimensionality respect to the original categorical bins
"""


class CVAE_CatEnergy_ContGeom_TaskAdaptive(keras.Model):
    """
    Continuous-geometry Task-Adaptive CVAE.

    Continuous targets:

        y_cont = [
            u_r_q,
            u_v_q,
            cphi_r,
            sphi_r,
            cphi_v,
            sphi_v,
        ]

    Categorical targets:

        energy_idx
    """

    def __init__(
        self,
        n_types,
        n_energy_bins,
        y_cont_dim=6,

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
        w_ur=1.0,
        w_uv=1.0,
        w_phi_r=1.0,
        w_phi_v=1.0,

        energy_sampling_temperature=1.0,

        phi_v_mse_weight=0.5,
        phi_v_ang_weight=1.2,

        stem_width=64,
        deep_decoder_hidden=(128,128,64),
        energy_branch_hidden=(48,48),
    ):
        super().__init__()

        self.n_types = int(n_types)
        self.n_energy_bins = int(n_energy_bins)
        self.y_cont_dim = int(y_cont_dim)

        self.latent_dim = latent_dim
        self.beta = beta

        self.type_weights = type_weights

        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.lambda_sigma = lambda_sigma
        self.sigma_target = sigma_target

        self.lambda_phi = lambda_phi
        self.lambda_phi_r = lambda_phi_r

        self.energy_sampling_temperature = energy_sampling_temperature

        self.phi_v_mse_weight = phi_v_mse_weight
        self.phi_v_ang_weight = phi_v_ang_weight

        self.task_weights = {
            "energy": float(w_energy),
            "ur": float(w_ur),
            "uv": float(w_uv),
            "phi_r": float(w_phi_r),
            "phi_v": float(w_phi_v),
        }

        # =====================================================
        # Encoder
        # =====================================================

        enc_in = layers.Input(
            shape=(
                self.y_cont_dim
                + self.n_energy_bins
                + self.n_types,
            )
        )

        x = enc_in

        for h in hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.05)(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)

        self.encoder = keras.Model(
            enc_in,
            [z_mean, z_logvar],
            name="encoder",
        )

        # =====================================================
        # Decoder stem
        # =====================================================

        dec_in = layers.Input(
            shape=(latent_dim + self.n_types,)
        )

        stem = layers.Dense(stem_width)(dec_in)
        stem = layers.LeakyReLU(0.03)(stem)

        self.decoder_stem = keras.Model(
            dec_in,
            stem,
            name="decoder_stem",
        )

        # =====================================================
        # Deep trunk
        # =====================================================

        trunk_in = layers.Input(shape=(stem_width,))

        x = trunk_in

        for h in deep_decoder_hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.03)(x)

        self.decoder_deep_trunk = keras.Model(
            trunk_in,
            x,
            name="decoder_deep_trunk",
        )

        # =====================================================
        # Energy branch
        # =====================================================

        energy_in = layers.Input(shape=(stem_width,))

        x = energy_in

        for h in energy_branch_hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.03)(x)

        self.energy_branch = keras.Model(
            energy_in,
            x,
            name="energy_branch",
        )

        # =====================================================
        # Position branch
        # =====================================================

        self.position_branch = keras.Sequential([
            layers.Input(shape=(deep_decoder_hidden[-1],)),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ])

        # =====================================================
        # Direction branch
        # =====================================================

        self.direction_branch = keras.Sequential([
            layers.Input(shape=(deep_decoder_hidden[-1],)),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ])

        # ======================================================
        # Decoder heads
        # ======================================================
        self.energy_logits_head = layers.Dense(
            self.n_energy_bins,
            name="energy_logits",
        )

        # u_r_q Gaussian head
        self.ur_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="ur_head")

        self.ur_mu_head = layers.Dense(1, name="ur_mu")
        self.ur_logsigma_head = layers.Dense(1, name="ur_logsigma")

        # u_v_q Gaussian head
        self.uv_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="uv_head")

        self.uv_mu_head = layers.Dense(1, name="uv_mu")
        self.uv_logsigma_head = layers.Dense(1, name="uv_logsigma")

        # phi_r head
        self.phi_r_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
            layers.Dense(2),
        ], name="phi_r_head")

        # phi_v head
        self.phi_v_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
            layers.Dense(2),
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
        self.phi_v_loss_tracker = keras.metrics.Mean(name="phi_v_loss")

        self.energy_ce_tracker = keras.metrics.Mean(name="energy_ce")

        self.ur_nll_tracker = keras.metrics.Mean(name="ur_nll")
        self.uv_nll_tracker = keras.metrics.Mean(name="uv_nll")

        self.phi_r_mse_tracker = keras.metrics.Mean(name="phi_r_mse")

        self.phi_v_mse_tracker = keras.metrics.Mean(name="phi_v_mse")
        self.phi_v_ang_tracker = keras.metrics.Mean(name="phi_v_ang")

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
            self.ur_nll_tracker,
            self.uv_nll_tracker,
            self.phi_r_mse_tracker,
            self.phi_v_mse_tracker,
            self.phi_v_ang_tracker,
            self.phi_v_loss_tracker,
        ]

    # ==========================================================
    # Task-weight API
    # ==========================================================
    def get_task_weight(self, task_name):
        return float(self.task_weights[task_name])

    def set_task_weight(self, task_name, value):
        value = float(value)
        self.task_weights[task_name] = value

        if task_name == "energy":
            self.w_energy = value
        elif task_name == "ur":
            self.w_ur = value
        elif task_name == "uv":
            self.w_uv = value
        elif task_name == "phi_r":
            self.w_phi_r = value
        elif task_name == "phi_v":
            self.w_phi_v = value
        else:
            raise ValueError(f"Unknown task name: {task_name}")

    def decay_task_weight(self, task_name, factor=0.5, min_value=0.0):
        old = self.get_task_weight(task_name)
        new = max(min_value, old * float(factor))
        self.set_task_weight(task_name, new)
        return old, new

    # ==========================================================
    # Utilities
    # ==========================================================
    def sample_z(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def _sigma_regularizer(self, params):

        if self.lambda_sigma <= 0:
            return tf.constant(0.0, dtype=tf.float32)

        reg_ur = tf.reduce_mean(
            tf.square(
                tf.nn.relu(
                    params["ur_logsigma"] - self.sigma_target
                )
            )
        )

        reg_uv = tf.reduce_mean(
            tf.square(
                tf.nn.relu(
                    params["uv_logsigma"] - self.sigma_target
                )
            )
        )

        return self.lambda_sigma * (reg_ur + reg_uv)

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


    # ==========================================================
    # Decoder
    # ==========================================================
    def _decode_params(self, z, cond, training=False):

        base = tf.concat([z, cond], axis=1)

        stem = self.decoder_stem(base, training=training)
        deep = self.decoder_deep_trunk(stem, training=training)

        energy_feat = self.energy_branch(stem, training=training)

        pos_feat = self.position_branch(deep, training=training)
        dir_feat = self.direction_branch(deep, training=training)

        energy_logits = self.energy_logits_head(
            energy_feat
        )

        ur_feat = self.ur_head(
            pos_feat,
            training=training,
        )

        ur_mu = self.ur_mu_head(ur_feat)

        ur_logsigma = tf.clip_by_value(
            self.ur_logsigma_head(ur_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        uv_feat = self.uv_head(
            dir_feat,
            training=training,
        )

        uv_mu = self.uv_mu_head(uv_feat)

        uv_logsigma = tf.clip_by_value(
            self.uv_logsigma_head(uv_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        phi_r = self.phi_r_head(
            pos_feat,
            training=training,
        )

        phi_v = self.phi_v_head(
            dir_feat,
            training=training,
        )

        return {
            "energy_logits": energy_logits,
            "ur_mu": ur_mu,
            "ur_logsigma": ur_logsigma,
            "uv_mu": uv_mu,
            "uv_logsigma": uv_logsigma,
            "phi_r": phi_r,
            "phi_v": phi_v,
        }

    def _reconstruction_terms(self, y_cont_true, E_idx_true, params,):  
        ur_true = y_cont_true[:,0:1]
        uv_true = y_cont_true[:,1:2]

        phi_r_true = y_cont_true[:,2:4]
        phi_v_true = y_cont_true[:,4:6]

        energy_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=E_idx_true,
            logits=params["energy_logits"]
        )

        ur_nll = tf.squeeze(
            gaussian_nll(ur_true, params["ur_mu"], params["ur_logsigma"]),
            axis=1
        )

        uv_nll = tf.squeeze(
            gaussian_nll(
                uv_true,
                params["uv_mu"],
                params["uv_logsigma"],
            ),
            axis=1,
        )

        phi_r_mse = tf.reduce_sum(tf.square(phi_r_true - params["phi_r"]), axis=1)

        phi_v_pred_unit = normalize_2d_pair(params["phi_v"])
        phi_v_mse = tf.reduce_sum(tf.square(phi_v_true - phi_v_pred_unit), axis=1)
        phi_v_ang = angular_loss_2d(phi_v_true, params["phi_v"])
        phi_v_loss = (
            self.phi_v_mse_weight * phi_v_mse +
            self.phi_v_ang_weight * phi_v_ang
        )


        rec_per = (
            self.task_weights["energy"] * energy_ce +
            self.task_weights["ur"]     * ur_nll +
            self.task_weights["uv"]     * uv_nll +
            self.task_weights["phi_r"]  * phi_r_mse +
            self.task_weights["phi_v"]  * phi_v_loss
        )

        pieces = {
            "energy_ce": energy_ce,
            "ur_nll": ur_nll,
            "uv_nll": uv_nll,
            "phi_r_mse": phi_r_mse,
            "phi_v_mse": phi_v_mse,
            "phi_v_ang": phi_v_ang,
            "phi_v_loss": phi_v_loss,
        }
        return rec_per, pieces

    # ==========================================================
    # Keras train/test
    # ==========================================================
    def train_step(self, data):
        (y_cont, E_idx_true, cond), _ = data

        E_onehot = tf.one_hot(
            E_idx_true,
            depth=self.n_energy_bins,
            dtype=tf.float32,
        )

        x_in = tf.concat([y_cont, E_onehot, cond], axis=1)

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(x_in, training=True)
            z = self.sample_z(z_mean, z_logvar)

            params = self._decode_params(z, cond, training=True)

            rec_per, pieces = self._reconstruction_terms(
                y_cont,
                E_idx_true,
                params,
            )

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
                axis=1,
            )
            kl = tf.reduce_mean(kl_per)

            sigma_reg = self._sigma_regularizer(params)
            phi_reg, phi_r_reg = self._phi_regularizers(params)

            loss = rec + self.beta * kl + sigma_reg + phi_reg + phi_r_reg

        grads = tape.gradient(loss, self.trainable_variables)

        grads_and_vars = []
        seen = set()

        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue

            vid = id(v)
            if vid in seen:
                continue

            seen.add(vid)
            grads_and_vars.append((g, v))

        self.optimizer.apply_gradients(grads_and_vars)

        self.loss_tracker.update_state(loss)
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        self.nll_tracker.update_state(nll)

        self.sigma_reg_tracker.update_state(sigma_reg)
        self.phi_reg_tracker.update_state(phi_reg)
        self.phi_r_reg_tracker.update_state(phi_r_reg)

        self.energy_ce_tracker.update_state(tf.reduce_mean(pieces["energy_ce"]))
        self.ur_nll_tracker.update_state(tf.reduce_mean(pieces["ur_nll"]))
        self.uv_nll_tracker.update_state(tf.reduce_mean(pieces["uv_nll"]))
        self.phi_r_mse_tracker.update_state(tf.reduce_mean(pieces["phi_r_mse"]))
        self.phi_v_mse_tracker.update_state(tf.reduce_mean(pieces["phi_v_mse"]))
        self.phi_v_ang_tracker.update_state(tf.reduce_mean(pieces["phi_v_ang"]))
        self.phi_v_loss_tracker.update_state(tf.reduce_mean(pieces["phi_v_loss"]))

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (y_cont, E_idx_true, cond), _ = data

        E_onehot = tf.one_hot(
            E_idx_true,
            depth=self.n_energy_bins,
            dtype=tf.float32,
        )

        x_in = tf.concat([y_cont, E_onehot, cond], axis=1)

        z_mean, z_logvar = self.encoder(x_in, training=False)
        z = self.sample_z(z_mean, z_logvar)

        params = self._decode_params(z, cond, training=False)

        rec_per, pieces = self._reconstruction_terms(
            y_cont,
            E_idx_true,
            params,
        )

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
            axis=1,
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
        self.ur_nll_tracker.update_state(tf.reduce_mean(pieces["ur_nll"]))
        self.uv_nll_tracker.update_state(tf.reduce_mean(pieces["uv_nll"]))
        self.phi_r_mse_tracker.update_state(tf.reduce_mean(pieces["phi_r_mse"]))
        self.phi_v_mse_tracker.update_state(tf.reduce_mean(pieces["phi_v_mse"]))
        self.phi_v_ang_tracker.update_state(tf.reduce_mean(pieces["phi_v_ang"]))
        self.phi_v_loss_tracker.update_state(tf.reduce_mean(pieces["phi_v_loss"]))

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

    def generate(self, cond, n_samples):
        z = tf.random.normal((n_samples, self.latent_dim))
        params = self.decode(z, cond)

        energy_idx = self._sample_energy_from_logits(params["energy_logits"])

        ur_eps = tf.random.normal(tf.shape(params["ur_mu"]))
        ur_q = params["ur_mu"] + tf.exp(params["ur_logsigma"]) * ur_eps

        uv_eps = tf.random.normal(tf.shape(params["uv_mu"]))
        uv_q = params["uv_mu"] + tf.exp(params["uv_logsigma"]) * uv_eps

        phi_r = params["phi_r"]
        phi_v = params["phi_v"]

        y_cont = tf.concat(
            [
                ur_q,
                uv_q,
                phi_r,
                phi_v,
            ],
            axis=1,
        )

        return {
            "energy_idx": energy_idx,
            "y_cont": y_cont,
            "params": params,
        }
    


class CVAE_CatEnergy_ContPhi_TaskAdaptive(keras.Model):
    """
    Continuous-geometry Task-Adaptive CVAE.

    Continuous targets:

        y_cont = [
            u_r_q,
            u_v_q,
            phi_r_q,
            phi_v_q,
        ]

    Categorical targets:

        energy_idx
    """

    def __init__(
        self,
        n_types,
        n_energy_bins,
        y_cont_dim=4,

        latent_dim=8,
        hidden=(128, 128, 64),

        beta=0.1,
        type_weights=None,

        min_log_sigma=-6.0,
        max_log_sigma=1.5,

        lambda_sigma=1e-3,
        sigma_target=-2.0,

        w_energy=1.0,
        w_ur=1.0,
        w_uv=1.0,
        w_phi_r=1.0,
        w_phi_v=1.0,

        energy_sampling_temperature=1.0,

        stem_width=64,
        deep_decoder_hidden=(128,128,64),
        energy_branch_hidden=(48,48),
    ):
        super().__init__()

        self.n_types = int(n_types)
        self.n_energy_bins = int(n_energy_bins)
        self.y_cont_dim = int(y_cont_dim)

        self.latent_dim = latent_dim
        self.beta = beta

        self.type_weights = type_weights

        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.lambda_sigma = lambda_sigma
        self.sigma_target = sigma_target


        self.energy_sampling_temperature = energy_sampling_temperature

        self.task_weights = {
            "energy": float(w_energy),
            "ur": float(w_ur),
            "uv": float(w_uv),
            "phi_r": float(w_phi_r),
            "phi_v": float(w_phi_v),
        }

        # =====================================================
        # Encoder
        # =====================================================

        enc_in = layers.Input(
            shape=(
                self.y_cont_dim
                + self.n_energy_bins
                + self.n_types,
            )
        )

        x = enc_in

        for h in hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.05)(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)

        self.encoder = keras.Model(
            enc_in,
            [z_mean, z_logvar],
            name="encoder",
        )

        # =====================================================
        # Decoder stem
        # =====================================================

        dec_in = layers.Input(
            shape=(latent_dim + self.n_types,)
        )

        stem = layers.Dense(stem_width)(dec_in)
        stem = layers.LeakyReLU(0.03)(stem)

        self.decoder_stem = keras.Model(
            dec_in,
            stem,
            name="decoder_stem",
        )

        # =====================================================
        # Deep trunk
        # =====================================================

        trunk_in = layers.Input(shape=(stem_width,))

        x = trunk_in

        for h in deep_decoder_hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.03)(x)

        self.decoder_deep_trunk = keras.Model(
            trunk_in,
            x,
            name="decoder_deep_trunk",
        )

        # =====================================================
        # Energy branch
        # =====================================================

        energy_in = layers.Input(shape=(stem_width,))

        x = energy_in

        for h in energy_branch_hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.03)(x)

        self.energy_branch = keras.Model(
            energy_in,
            x,
            name="energy_branch",
        )

        # =====================================================
        # Position branch
        # =====================================================

        self.position_branch = keras.Sequential([
            layers.Input(shape=(deep_decoder_hidden[-1],)),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ])

        # =====================================================
        # Direction branch
        # =====================================================

        self.direction_branch = keras.Sequential([
            layers.Input(shape=(deep_decoder_hidden[-1],)),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ])

        # ======================================================
        # Decoder heads
        # ======================================================
        self.energy_logits_head = layers.Dense(
            self.n_energy_bins,
            name="energy_logits",
        )

        # u_r_q Gaussian head
        self.ur_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="ur_head")

        self.ur_mu_head = layers.Dense(1, name="ur_mu")
        self.ur_logsigma_head = layers.Dense(1, name="ur_logsigma")

        # u_v_q Gaussian head
        self.uv_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="uv_head")

        self.uv_mu_head = layers.Dense(1, name="uv_mu")
        self.uv_logsigma_head = layers.Dense(1, name="uv_logsigma")

        # phi_r_q Gaussian head
        self.phi_r_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="phi_r_head")

        self.phi_r_mu_head = layers.Dense(1, name="phi_r_mu")
        self.phi_r_logsigma_head = layers.Dense(1, name="phi_r_logsigma")


        # phi_v_q Gaussian head
        self.phi_v_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="phi_v_head")

        self.phi_v_mu_head = layers.Dense(1, name="phi_v_mu")
        self.phi_v_logsigma_head = layers.Dense(1, name="phi_v_logsigma")

        # ======================================================
        # Metrics
        # ======================================================
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.rec_tracker = keras.metrics.Mean(name="rec")
        self.kl_tracker = keras.metrics.Mean(name="kl")

        self.nll_tracker = keras.metrics.Mean(name="nll")
        self.sigma_reg_tracker = keras.metrics.Mean(name="sigma_reg")


        self.energy_ce_tracker = keras.metrics.Mean(name="energy_ce")

        self.ur_nll_tracker = keras.metrics.Mean(name="ur_nll")
        self.uv_nll_tracker = keras.metrics.Mean(name="uv_nll")

        self.phi_r_nll_tracker = keras.metrics.Mean(name="phi_r_nll")
        self.phi_v_nll_tracker = keras.metrics.Mean(name="phi_v_nll")


    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.rec_tracker,
            self.kl_tracker,
            self.nll_tracker,
            self.sigma_reg_tracker,

            self.energy_ce_tracker,
            self.ur_nll_tracker,
            self.uv_nll_tracker,
            self.phi_r_nll_tracker,
            self.phi_v_nll_tracker,
        ]

    # ==========================================================
    # Task-weight API
    # ==========================================================
    def get_task_weight(self, task_name):
        return float(self.task_weights[task_name])

    def set_task_weight(self, task_name, value):
        value = float(value)
        self.task_weights[task_name] = value

        if task_name == "energy":
            self.w_energy = value
        elif task_name == "ur":
            self.w_ur = value
        elif task_name == "uv":
            self.w_uv = value
        elif task_name == "phi_r":
            self.w_phi_r = value
        elif task_name == "phi_v":
            self.w_phi_v = value
        else:
            raise ValueError(f"Unknown task name: {task_name}")

    def decay_task_weight(self, task_name, factor=0.5, min_value=0.0):
        old = self.get_task_weight(task_name)
        new = max(min_value, old * float(factor))
        self.set_task_weight(task_name, new)
        return old, new

    # ==========================================================
    # Utilities
    # ==========================================================
    def sample_z(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def _sigma_regularizer(self, params):

        if self.lambda_sigma <= 0:
            return tf.constant(0.0, dtype=tf.float32)

        regs = []

        for key in [
            "ur_logsigma",
            "uv_logsigma",
            "phi_r_logsigma",
            "phi_v_logsigma",
        ]:
            regs.append(
                tf.reduce_mean(
                    tf.square(
                        tf.nn.relu(params[key] - self.sigma_target)
                    )
                )
            )

        return self.lambda_sigma * tf.add_n(regs)



    # ==========================================================
    # Decoder
    # ==========================================================
    def _decode_params(self, z, cond, training=False):

        base = tf.concat([z, cond], axis=1)

        stem = self.decoder_stem(base, training=training)
        deep = self.decoder_deep_trunk(stem, training=training)

        energy_feat = self.energy_branch(stem, training=training)

        pos_feat = self.position_branch(deep, training=training)
        dir_feat = self.direction_branch(deep, training=training)

        energy_logits = self.energy_logits_head(
            energy_feat
        )

        ur_feat = self.ur_head(
            pos_feat,
            training=training,
        )

        ur_mu = self.ur_mu_head(ur_feat)

        ur_logsigma = tf.clip_by_value(
            self.ur_logsigma_head(ur_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        uv_feat = self.uv_head(
            dir_feat,
            training=training,
        )

        uv_mu = self.uv_mu_head(uv_feat)

        uv_logsigma = tf.clip_by_value(
            self.uv_logsigma_head(uv_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        phi_r_feat = self.phi_r_head(
            pos_feat,
            training=training,
        )

        phi_r_mu = self.phi_r_mu_head(phi_r_feat)

        phi_r_logsigma = tf.clip_by_value(
            self.phi_r_logsigma_head(phi_r_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        phi_v_feat = self.phi_v_head(
            dir_feat,
            training=training,
        )

        phi_v_mu = self.phi_v_mu_head(phi_v_feat)

        phi_v_logsigma = tf.clip_by_value(
            self.phi_v_logsigma_head(phi_v_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        return {
            "energy_logits": energy_logits,

            "ur_mu": ur_mu,
            "ur_logsigma": ur_logsigma,

            "uv_mu": uv_mu,
            "uv_logsigma": uv_logsigma,

            "phi_r_mu": phi_r_mu,
            "phi_r_logsigma": phi_r_logsigma,

            "phi_v_mu": phi_v_mu,
            "phi_v_logsigma": phi_v_logsigma,
        }

    def _reconstruction_terms(self, y_cont_true, E_idx_true, params):

        ur_true = y_cont_true[:, 0:1]
        uv_true = y_cont_true[:, 1:2]
        phi_r_true = y_cont_true[:, 2:3]
        phi_v_true = y_cont_true[:, 3:4]

        energy_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=E_idx_true,
            logits=params["energy_logits"],
        )

        ur_nll = tf.squeeze(
            gaussian_nll(ur_true, params["ur_mu"], params["ur_logsigma"]),
            axis=1,
        )

        uv_nll = tf.squeeze(
            gaussian_nll(uv_true, params["uv_mu"], params["uv_logsigma"]),
            axis=1,
        )

        phi_r_nll = tf.squeeze(
            gaussian_nll(
                phi_r_true,
                params["phi_r_mu"],
                params["phi_r_logsigma"],
            ),
            axis=1,
        )

        phi_v_nll = tf.squeeze(
            gaussian_nll(
                phi_v_true,
                params["phi_v_mu"],
                params["phi_v_logsigma"],
            ),
            axis=1,
        )

        rec_per = (
            self.task_weights["energy"] * energy_ce +
            self.task_weights["ur"]     * ur_nll +
            self.task_weights["uv"]     * uv_nll +
            self.task_weights["phi_r"]  * phi_r_nll +
            self.task_weights["phi_v"]  * phi_v_nll
        )

        pieces = {
            "energy_ce": energy_ce,
            "ur_nll": ur_nll,
            "uv_nll": uv_nll,
            "phi_r_nll": phi_r_nll,
            "phi_v_nll": phi_v_nll,
        }

        return rec_per, pieces

    # ==========================================================
    # Keras train/test
    # ==========================================================
    def train_step(self, data):
        (y_cont, E_idx_true, cond), _ = data

        E_onehot = tf.one_hot(
            E_idx_true,
            depth=self.n_energy_bins,
            dtype=tf.float32,
        )

        x_in = tf.concat([y_cont, E_onehot, cond], axis=1)

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(x_in, training=True)
            z = self.sample_z(z_mean, z_logvar)

            params = self._decode_params(z, cond, training=True)

            rec_per, pieces = self._reconstruction_terms(
                y_cont,
                E_idx_true,
                params,
            )

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
                axis=1,
            )
            kl = tf.reduce_mean(kl_per)

            sigma_reg = self._sigma_regularizer(params)

            loss = rec + self.beta * kl + sigma_reg

        grads = tape.gradient(loss, self.trainable_variables)

        grads_and_vars = []
        seen = set()

        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue

            vid = id(v)
            if vid in seen:
                continue

            seen.add(vid)
            grads_and_vars.append((g, v))

        self.optimizer.apply_gradients(grads_and_vars)

        self.loss_tracker.update_state(loss)
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        self.nll_tracker.update_state(nll)

        self.sigma_reg_tracker.update_state(sigma_reg)

        self.energy_ce_tracker.update_state(tf.reduce_mean(pieces["energy_ce"]))
        self.ur_nll_tracker.update_state(tf.reduce_mean(pieces["ur_nll"]))
        self.uv_nll_tracker.update_state(tf.reduce_mean(pieces["uv_nll"]))
        self.phi_r_nll_tracker.update_state(tf.reduce_mean(pieces["phi_r_nll"]))
        self.phi_v_nll_tracker.update_state(tf.reduce_mean(pieces["phi_v_nll"]))

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (y_cont, E_idx_true, cond), _ = data

        E_onehot = tf.one_hot(
            E_idx_true,
            depth=self.n_energy_bins,
            dtype=tf.float32,
        )

        x_in = tf.concat([y_cont, E_onehot, cond], axis=1)

        z_mean, z_logvar = self.encoder(x_in, training=False)
        z = self.sample_z(z_mean, z_logvar)

        params = self._decode_params(z, cond, training=False)

        rec_per, pieces = self._reconstruction_terms(
            y_cont,
            E_idx_true,
            params,
        )

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
            axis=1,
        )
        kl = tf.reduce_mean(kl_per)

        sigma_reg = self._sigma_regularizer(params)

        loss = rec + self.beta * kl + sigma_reg

        self.loss_tracker.update_state(loss)
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        self.nll_tracker.update_state(nll)

        self.sigma_reg_tracker.update_state(sigma_reg)

        self.energy_ce_tracker.update_state(tf.reduce_mean(pieces["energy_ce"]))
        self.ur_nll_tracker.update_state(tf.reduce_mean(pieces["ur_nll"]))
        self.uv_nll_tracker.update_state(tf.reduce_mean(pieces["uv_nll"]))
        self.phi_r_nll_tracker.update_state(tf.reduce_mean(pieces["phi_r_nll"]))
        self.phi_v_nll_tracker.update_state(tf.reduce_mean(pieces["phi_v_nll"]))

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

    def generate(self, cond, n_samples):
        z = tf.random.normal((n_samples, self.latent_dim))
        params = self.decode(z, cond)

        energy_idx = self._sample_energy_from_logits(params["energy_logits"])

        ur_eps = tf.random.normal(tf.shape(params["ur_mu"]))
        ur_q = params["ur_mu"] + tf.exp(params["ur_logsigma"]) * ur_eps

        uv_eps = tf.random.normal(tf.shape(params["uv_mu"]))
        uv_q = params["uv_mu"] + tf.exp(params["uv_logsigma"]) * uv_eps

        phi_r_eps = tf.random.normal(tf.shape(params["phi_r_mu"]))
        phi_r_q = params["phi_r_mu"] + tf.exp(params["phi_r_logsigma"]) * phi_r_eps

        phi_v_eps = tf.random.normal(tf.shape(params["phi_v_mu"]))
        phi_v_q = params["phi_v_mu"] + tf.exp(params["phi_v_logsigma"]) * phi_v_eps

        y_cont = tf.concat(
            [
                ur_q,
                uv_q,
                phi_r_q,
                phi_v_q,
            ],
            axis=1,
        )

        return {
            "energy_idx": energy_idx,
            "y_cont": y_cont,
            "params": params,
        }

class CVAE_MixEnergy_ContPhi_TaskAdaptive(keras.Model):
    """
    Continuous-geometry Task-Adaptive CVAE with a continuum + fixed-line
    Gaussian mixture energy head (v0.8), copied and modified from
    CVAE_CatEnergy_ContPhi_TaskAdaptive - v0.7.2 is untouched.

    Continuous targets:

        y_cont = [
            u_r_q,
            u_v_q,
            phi_r_q,
            phi_v_q,
            energy_y,
            gate_target_0, ..., gate_target_n_lines,   # see below
        ]

    Energy model, in a transformed energy space y = T(E):

        p(y | z, cond) = sum_{k=1}^{K} pi_k * N(y; mu_k(h), sigma_k(h))
                       + sum_l pi_l * N(y; line_positions_y[l], line_sigma)

    The K continuum sub-components (mu_k, sigma_k) and the gate (pi, over
    K+n_lines slots) are learned per-sample; the L line positions are fixed
    physics inputs (already mapped into y-space outside the model); the
    line width is a single learned scalar shared across all lines (no
    detector-resolution constant is available to fix it instead - see
    docs/v0.8_beta_plan.md). K = n_continuum_components, default 1 (a
    single Gaussian continuum, the original v0.8 Part 2/3 design). K>1
    (docs/v0.8_fixing_plan.md Task #25) lets the continuum itself represent
    multi-modal or sharply-peaked real shapes that a single Gaussian can't
    - confirmed empirically that a single continuum component forces a
    trade-off between suppressing spurious density near lines (needs a
    tightly-constrained, low-flexibility mean) and fitting a sharply peaked
    real continuum like CryoSphere-Small's (needs a precisely-localized,
    higher-flexibility mean) that can't be resolved by regularizer tuning
    alone on one Gaussian.

    Auxiliary gate supervision: the true energy is both an encoder input
    (part of y_cont) and a decoder reconstruction target, so without
    supervision the encoder can leak "which line this event is near" into
    z and let the continuum mean chase it - the gate then never learns to
    use the line components at all (confirmed empirically before this was
    added). To break that degeneracy, y_cont carries n_lines+1 extra
    "gate_target" columns (built by data.preprocessing.build_gate_targets
    from physical proximity to the known line positions, independent of
    anything the model learns) that a small auxiliary cross-entropy loss
    pulls softmax(gate_logits) toward. y_cont_dim is derived automatically
    from n_lines - there is no free-standing y_cont_dim argument.

    Continuum-repulsion regularizer: auxiliary gate supervision only
    constrains the discrete gate - nothing stops the continuum's own
    per-sample mean energy_cont_mu(z) from ALSO drifting toward a line's
    position for latent codes the encoder assigns to line-proximal
    training events (the same z-leakage degeneracy, now on the continuum
    side; confirmed empirically on real data as spurious density appearing
    exactly at known line locations even with gate supervision in place).
    w_continuum_repulsion/continuum_repulsion_margin penalize
    energy_cont_mu for landing within `continuum_repulsion_margin` of any
    line_positions_y entry, forcing the (already gate-supervised) discrete
    line components to exclusively own that density instead.

    Continuum-balance regularizer (docs/v0.8_fixing_plan.md Task #26):
    confirmed empirically that naively setting n_continuum_components>1
    doesn't reliably give a useful K-way split - with no supervision
    telling the K continuum slots how to divide responsibility (unlike the
    lines, which get build_gate_targets), and near-identical zero-biased
    initial outputs giving optimization no reason to break symmetry, one
    sub-component can absorb almost all continuum mass while the others
    collapse to near-zero usage, which strictly hurt shape fidelity
    relative to K=1 in testing. w_continuum_balance penalizes the batch-mean
    usage of the K continuum slots (renormalized to sum to 1 among
    themselves) for deviating from uniform (1/K each); continuum_mu_init
    spreads the K sub-components' initial means instead of leaving them all
    at 0, so they start in different parts of y-space rather than
    competing from an identical starting point. Both are no-ops when
    n_continuum_components == 1.
    """

    def __init__(
        self,
        n_types,
        line_positions_y,

        latent_dim=8,
        hidden=(128, 128, 64),

        beta=0.2,
        type_weights=None,

        min_log_sigma=-6.0,
        max_log_sigma=1.5,

        lambda_sigma=1e-3,
        sigma_target=-2.0,

        w_energy=1.0,
        w_ur=1.0,
        w_uv=1.0,
        w_phi_r=1.0,
        w_phi_v=1.0,
        w_gate_aux=0.3,
        w_continuum_repulsion=0.3,
        continuum_repulsion_margin=0.05,
        n_continuum_components=1,
        w_continuum_balance=1.0,
        continuum_mu_init=None,

        continuum_mode="gaussian",
        energy_flow_condition="z_cond",
        continuum_flow_bins=8,
        continuum_flow_transforms=2,
        continuum_flow_interval=5.0,
        continuum_flow_conditioner_hidden=(64,),
        continuum_flow_y_mean=None,
        continuum_flow_y_scale=None,

        energy_sampling_temperature=1.0,
        line_logsigma_init=-2.0,
        line_logsigma_trainable=True,

        stem_width=64,
        deep_decoder_hidden=(128,128,64),
        energy_branch_hidden=(48,48),
        energy_cont_head_hidden=(64,32),
    ):
        super().__init__()

        self.n_types = int(n_types)

        line_positions_y = np.asarray(line_positions_y, dtype=np.float32)
        self.n_lines = int(line_positions_y.shape[0])
        self.line_positions_y = tf.constant(line_positions_y, dtype=tf.float32)

        if continuum_mode not in ("gaussian", "flow"):
            raise ValueError(
                f"continuum_mode must be 'gaussian' or 'flow', got {continuum_mode!r}."
            )
        self.continuum_mode = str(continuum_mode)

        if energy_flow_condition not in ("z_cond", "cond"):
            raise ValueError(
                "energy_flow_condition must be 'z_cond' or 'cond', got "
                f"{energy_flow_condition!r}."
            )
        self.energy_flow_condition = str(energy_flow_condition)

        # The flow is a single continuum slot in the gate (indices 0..n_lines
        # stay {continuum, line_1..line_L}); the n_continuum_components knob
        # only applies to the Gaussian sub-mixture. Force 1 in flow mode so
        # the gate width, gate-aux pooling, and the metrics/generate slot
        # convention (n_continuum_components + i == line i) all stay correct.
        if self.continuum_mode == "flow":
            self.n_continuum_components = 1
        else:
            self.n_continuum_components = int(n_continuum_components)

        # 4 geometry columns + 1 energy_y + (n_lines + 1) gate-target columns.
        self.y_cont_dim = 4 + 1 + (self.n_lines + 1)

        self.w_gate_aux = float(w_gate_aux)
        self.w_continuum_repulsion = float(w_continuum_repulsion)
        self.continuum_repulsion_margin = float(continuum_repulsion_margin)
        self.w_continuum_balance = float(w_continuum_balance)

        if continuum_mu_init is None:
            if self.n_continuum_components > 1:
                continuum_mu_init = np.linspace(-1.0, 1.0, self.n_continuum_components)
            else:
                continuum_mu_init = np.zeros(self.n_continuum_components)
        else:
            continuum_mu_init = np.asarray(continuum_mu_init, dtype=np.float32)
            if continuum_mu_init.shape[0] != self.n_continuum_components:
                raise ValueError(
                    "continuum_mu_init must have length n_continuum_components "
                    f"({self.n_continuum_components}), got shape {continuum_mu_init.shape}."
                )
        self.continuum_mu_init = continuum_mu_init.astype(np.float32)

        self.continuum_flow_bins = int(continuum_flow_bins)
        self.continuum_flow_transforms = int(continuum_flow_transforms)
        self.continuum_flow_interval = float(continuum_flow_interval)
        self.continuum_flow_conditioner_hidden = tuple(continuum_flow_conditioner_hidden)
        self.continuum_flow_y_mean = (
            0.0 if continuum_flow_y_mean is None else float(continuum_flow_y_mean)
        )
        self.continuum_flow_y_scale = (
            1.0 if continuum_flow_y_scale is None else float(continuum_flow_y_scale)
        )

        self.latent_dim = latent_dim
        self.beta = beta

        self.type_weights = type_weights

        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.lambda_sigma = lambda_sigma
        self.sigma_target = sigma_target


        self.energy_sampling_temperature = energy_sampling_temperature

        self.task_weights = {
            "energy": float(w_energy),
            "ur": float(w_ur),
            "uv": float(w_uv),
            "phi_r": float(w_phi_r),
            "phi_v": float(w_phi_v),
        }

        # =====================================================
        # Encoder
        # =====================================================

        enc_in = layers.Input(
            shape=(
                self.y_cont_dim
                + self.n_types,
            )
        )

        x = enc_in

        for h in hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.05)(x)

        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)

        self.encoder = keras.Model(
            enc_in,
            [z_mean, z_logvar],
            name="encoder",
        )

        # =====================================================
        # Decoder stem
        # =====================================================

        dec_in = layers.Input(
            shape=(latent_dim + self.n_types,)
        )

        stem = layers.Dense(stem_width)(dec_in)
        stem = layers.LeakyReLU(0.03)(stem)

        self.decoder_stem = keras.Model(
            dec_in,
            stem,
            name="decoder_stem",
        )

        # =====================================================
        # Deep trunk
        # =====================================================

        trunk_in = layers.Input(shape=(stem_width,))

        x = trunk_in

        for h in deep_decoder_hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.03)(x)

        self.decoder_deep_trunk = keras.Model(
            trunk_in,
            x,
            name="decoder_deep_trunk",
        )

        # =====================================================
        # Energy branch
        # =====================================================

        energy_in = layers.Input(shape=(stem_width,))

        x = energy_in

        for h in energy_branch_hidden:
            x = layers.Dense(h)(x)
            x = layers.LeakyReLU(0.03)(x)

        self.energy_branch = keras.Model(
            energy_in,
            x,
            name="energy_branch",
        )

        # =====================================================
        # Position branch
        # =====================================================

        self.position_branch = keras.Sequential([
            layers.Input(shape=(deep_decoder_hidden[-1],)),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ])

        # =====================================================
        # Direction branch
        # =====================================================

        self.direction_branch = keras.Sequential([
            layers.Input(shape=(deep_decoder_hidden[-1],)),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(128),
            layers.LeakyReLU(0.03),
            layers.Dense(64),
            layers.LeakyReLU(0.03),
        ])

        # ======================================================
        # Decoder heads
        # ======================================================

        # Energy mixture head: gate (n_continuum_components + n_lines slots)
        # and continuum mu/logsigma (n_continuum_components each), off the
        # (shallow) energy branch - same branch point as the old categorical
        # head, energy stays the "easy" task. Line positions are fixed
        # constants (set above); line width is a single learned scalar
        # shared across all lines.
        self.energy_gate_head = layers.Dense(
            self.n_continuum_components + self.n_lines,
            name="energy_gate_logits",
        )

        if self.continuum_mode == "flow":
            # Conditional normalizing-flow continuum: replaces the parametric
            # (mu, sigma) continuum component with a 1-D conditional RQS flow
            # over energy_y. See core/flows.py and docs/v0.8_fixing_plan.md
            # section 3(b).
            #
            # What the flow (and the gate) condition on is set by
            # energy_flow_condition:
            #   "z_cond" (default): condition on a function of the per-event
            #       latent z (via energy_cont_feat) and cond. This PRESERVES
            #       the energy<->geometry coupling that is a fundamental
            #       feature of the data (a given energy - especially a
            #       material-dependent fluorescence line - is correlated with
            #       the volume/geometry it came from, all mediated by z). This
            #       is the physically required default. NOTE: a z-conditioned
            #       flexible flow re-exposes the aggregated-posterior / prior
            #       mismatch (docs/v0.8_IAF_integration_plan.md sections 2,4) -
            #       generation samples z ~ N(0,I) which does not match the
            #       aggregated posterior - so faithful *generation* needs a
            #       learnable prior p(z|cond) (IAF-plan Phase 2), still TODO.
            #   "cond": condition only on cond (particle type). Removes the
            #       leak and made the single-peak synthetic test pass at
            #       convergence, but DROPS the energy<->geometry coupling, so
            #       it is an ablation / shape-capacity demonstration only, not
            #       the physically-correct default. (The synthetic stress test
            #       uses noise geometry, so it can only ever measure the leak,
            #       never the legitimate coupling - it cannot validate
            #       "z_cond" on its own.)
            if self.energy_flow_condition == "cond":
                flow_feat_dim = self.n_types
                self.energy_cont_head = None
            else:
                flow_feat_dim = energy_cont_head_hidden[-1]
                self.energy_cont_head = keras.Sequential([
                    layers.Input(shape=(energy_branch_hidden[-1],)),
                    layers.Dense(energy_cont_head_hidden[0]),
                    layers.LeakyReLU(0.05),
                    layers.Dense(energy_cont_head_hidden[1]),
                    layers.LeakyReLU(0.05),
                ], name="energy_cont_head")

            self.continuum_flow = ConditionalRQSFlow(
                feat_dim=flow_feat_dim,
                y_mean=self.continuum_flow_y_mean,
                y_scale=self.continuum_flow_y_scale,
                n_bins=self.continuum_flow_bins,
                n_transforms=self.continuum_flow_transforms,
                interval_half_width=self.continuum_flow_interval,
                conditioner_hidden=self.continuum_flow_conditioner_hidden,
                name="continuum_flow",
            )
            self.energy_cont_mu_head = None
            self.energy_cont_logsigma_head = None
        else:
            self.energy_cont_head = keras.Sequential([
                layers.Input(shape=(energy_branch_hidden[-1],)),
                layers.Dense(energy_cont_head_hidden[0]),
                layers.LeakyReLU(0.05),
                layers.Dense(energy_cont_head_hidden[1]),
                layers.LeakyReLU(0.05),
            ], name="energy_cont_head")
            self.continuum_flow = None
            self.energy_cont_mu_head = layers.Dense(
                self.n_continuum_components,
                name="energy_cont_mu",
                bias_initializer=keras.initializers.Constant(self.continuum_mu_init),
            )
            self.energy_cont_logsigma_head = layers.Dense(
                self.n_continuum_components, name="energy_cont_logsigma"
            )

        # When trainable, the mixture NLL tends to WIDEN the lines so they can
        # absorb nearby continuum density - which inflates their gate weight
        # and over-generates the lines. If the detector resolution is known,
        # pin the width (line_logsigma_trainable=False, line_logsigma_init =
        # log(resolution_sigma_in_y)) so each line can only explain events at
        # its own position and the continuum term must own the rest.
        self.line_logsigma = self.add_weight(
            name="line_logsigma",
            shape=(),
            initializer=keras.initializers.Constant(line_logsigma_init),
            trainable=bool(line_logsigma_trainable),
        )

        # u_r_q Gaussian head
        self.ur_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="ur_head")

        self.ur_mu_head = layers.Dense(1, name="ur_mu")
        self.ur_logsigma_head = layers.Dense(1, name="ur_logsigma")

        # u_v_q Gaussian head
        self.uv_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="uv_head")

        self.uv_mu_head = layers.Dense(1, name="uv_mu")
        self.uv_logsigma_head = layers.Dense(1, name="uv_logsigma")

        # phi_r_q Gaussian head
        self.phi_r_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="phi_r_head")

        self.phi_r_mu_head = layers.Dense(1, name="phi_r_mu")
        self.phi_r_logsigma_head = layers.Dense(1, name="phi_r_logsigma")


        # phi_v_q Gaussian head
        self.phi_v_head = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128),
            layers.LeakyReLU(0.05),
            layers.Dense(64),
            layers.LeakyReLU(0.05),
        ], name="phi_v_head")

        self.phi_v_mu_head = layers.Dense(1, name="phi_v_mu")
        self.phi_v_logsigma_head = layers.Dense(1, name="phi_v_logsigma")

        # ======================================================
        # Metrics
        # ======================================================
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.rec_tracker = keras.metrics.Mean(name="rec")
        self.kl_tracker = keras.metrics.Mean(name="kl")

        self.nll_tracker = keras.metrics.Mean(name="nll")
        self.sigma_reg_tracker = keras.metrics.Mean(name="sigma_reg")


        self.energy_mixture_nll_tracker = keras.metrics.Mean(name="energy_mixture_nll")
        self.gate_aux_loss_tracker = keras.metrics.Mean(name="gate_aux_loss")
        self.continuum_repulsion_tracker = keras.metrics.Mean(name="continuum_repulsion")
        self.continuum_balance_tracker = keras.metrics.Mean(name="continuum_balance")

        self.ur_nll_tracker = keras.metrics.Mean(name="ur_nll")
        self.uv_nll_tracker = keras.metrics.Mean(name="uv_nll")

        self.phi_r_nll_tracker = keras.metrics.Mean(name="phi_r_nll")
        self.phi_v_nll_tracker = keras.metrics.Mean(name="phi_v_nll")


    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.rec_tracker,
            self.kl_tracker,
            self.nll_tracker,
            self.sigma_reg_tracker,

            self.energy_mixture_nll_tracker,
            self.gate_aux_loss_tracker,
            self.continuum_repulsion_tracker,
            self.continuum_balance_tracker,
            self.ur_nll_tracker,
            self.uv_nll_tracker,
            self.phi_r_nll_tracker,
            self.phi_v_nll_tracker,
        ]

    # ==========================================================
    # Task-weight API
    # ==========================================================
    def get_task_weight(self, task_name):
        return float(self.task_weights[task_name])

    def set_task_weight(self, task_name, value):
        value = float(value)
        self.task_weights[task_name] = value

        if task_name == "energy":
            self.w_energy = value
        elif task_name == "ur":
            self.w_ur = value
        elif task_name == "uv":
            self.w_uv = value
        elif task_name == "phi_r":
            self.w_phi_r = value
        elif task_name == "phi_v":
            self.w_phi_v = value
        else:
            raise ValueError(f"Unknown task name: {task_name}")

    def decay_task_weight(self, task_name, factor=0.5, min_value=0.0):
        old = self.get_task_weight(task_name)
        new = max(min_value, old * float(factor))
        self.set_task_weight(task_name, new)
        return old, new

    # ==========================================================
    # Utilities
    # ==========================================================
    def sample_z(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def _line_logsigma_clipped(self):
        return tf.clip_by_value(
            self.line_logsigma, self.min_log_sigma, self.max_log_sigma
        )

    def _sigma_regularizer(self, params):

        if self.lambda_sigma <= 0:
            return tf.constant(0.0, dtype=tf.float32)

        regs = []

        sigma_keys = [
            "ur_logsigma",
            "uv_logsigma",
            "phi_r_logsigma",
            "phi_v_logsigma",
        ]
        # In flow mode the continuum has no explicit logsigma output (its
        # spread lives inside the flow), so there is nothing to regularize
        # for the continuum here.
        if self.continuum_mode == "gaussian":
            sigma_keys = ["energy_cont_logsigma"] + sigma_keys

        for key in sigma_keys:
            regs.append(
                tf.reduce_mean(
                    tf.square(
                        tf.nn.relu(params[key] - self.sigma_target)
                    )
                )
            )

        regs.append(
            tf.square(tf.nn.relu(self._line_logsigma_clipped() - self.sigma_target))
        )

        return self.lambda_sigma * tf.add_n(regs)



    # ==========================================================
    # Decoder
    # ==========================================================
    def _decode_params(self, z, cond, training=False):

        base = tf.concat([z, cond], axis=1)

        stem = self.decoder_stem(base, training=training)
        deep = self.decoder_deep_trunk(stem, training=training)

        energy_feat = self.energy_branch(stem, training=training)

        pos_feat = self.position_branch(deep, training=training)
        dir_feat = self.direction_branch(deep, training=training)

        if self.continuum_mode == "flow" and self.energy_flow_condition == "cond":
            # Ablation: gate and flow both condition on cond only (drops the
            # energy<->geometry coupling; see the flow-head construction
            # comment). Supervised gate over cond, no z-dependence.
            energy_gate_logits = self.energy_gate_head(cond)
            flow_cond = cond
        elif self.continuum_mode == "flow":
            # Default "z_cond": gate on z (via energy_feat) and the flow on a
            # function of z (energy_cont_feat) - preserves energy<->geometry
            # coupling.
            energy_gate_logits = self.energy_gate_head(energy_feat)
            flow_cond = self.energy_cont_head(energy_feat, training=training)
        else:
            energy_gate_logits = self.energy_gate_head(energy_feat)
            energy_cont_feat = self.energy_cont_head(energy_feat, training=training)

        ur_feat = self.ur_head(
            pos_feat,
            training=training,
        )

        ur_mu = self.ur_mu_head(ur_feat)

        ur_logsigma = tf.clip_by_value(
            self.ur_logsigma_head(ur_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        uv_feat = self.uv_head(
            dir_feat,
            training=training,
        )

        uv_mu = self.uv_mu_head(uv_feat)

        uv_logsigma = tf.clip_by_value(
            self.uv_logsigma_head(uv_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        phi_r_feat = self.phi_r_head(
            pos_feat,
            training=training,
        )

        phi_r_mu = self.phi_r_mu_head(phi_r_feat)

        phi_r_logsigma = tf.clip_by_value(
            self.phi_r_logsigma_head(phi_r_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        phi_v_feat = self.phi_v_head(
            dir_feat,
            training=training,
        )

        phi_v_mu = self.phi_v_mu_head(phi_v_feat)

        phi_v_logsigma = tf.clip_by_value(
            self.phi_v_logsigma_head(phi_v_feat),
            self.min_log_sigma,
            self.max_log_sigma,
        )

        out = {
            "energy_gate_logits": energy_gate_logits,

            "ur_mu": ur_mu,
            "ur_logsigma": ur_logsigma,

            "uv_mu": uv_mu,
            "uv_logsigma": uv_logsigma,

            "phi_r_mu": phi_r_mu,
            "phi_r_logsigma": phi_r_logsigma,

            "phi_v_mu": phi_v_mu,
            "phi_v_logsigma": phi_v_logsigma,
        }

        if self.continuum_mode == "flow":
            out["flow_cond"] = flow_cond
        else:
            out["energy_cont_mu"] = self.energy_cont_mu_head(energy_cont_feat)
            out["energy_cont_logsigma"] = tf.clip_by_value(
                self.energy_cont_logsigma_head(energy_cont_feat),
                self.min_log_sigma,
                self.max_log_sigma,
            )

        return out

    def _mixture_components(self, params):
        """Assemble (comp_mu, comp_logsigma), shape (batch, n_continuum_components
        + n_lines) each: the first n_continuum_components columns are the
        learned continuum sub-components, the remaining n_lines columns are
        the fixed line positions broadcast to the batch, with the shared
        learned line width."""
        batch = tf.shape(params["energy_cont_mu"])[0]

        line_logsigma_c = self._line_logsigma_clipped()
        line_mu_b = tf.broadcast_to(
            self.line_positions_y[None, :], (batch, self.n_lines)
        )
        line_logsigma_b = tf.broadcast_to(
            tf.reshape(line_logsigma_c, (1, 1)), (batch, self.n_lines)
        )

        comp_mu = tf.concat([params["energy_cont_mu"], line_mu_b], axis=1)
        comp_logsigma = tf.concat(
            [params["energy_cont_logsigma"], line_logsigma_b], axis=1
        )

        return comp_mu, comp_logsigma

    def _continuum_repulsion(self, params):
        """Penalize every continuum sub-component's mean for landing within
        continuum_repulsion_margin of any fixed line position - stops the
        continuum from encroaching on line territory via the same z-leakage
        degeneracy the auxiliary gate loss fixes for the gate (see class
        docstring). dist shape (batch, n_continuum_components, n_lines).

        No-op in flow mode: the flow has no single per-sample mean to repel,
        and its own flexibility plus the auxiliary gate supervision are the
        principled substitute for keeping continuum density off the lines."""
        if self.continuum_mode == "flow":
            return tf.constant(0.0, dtype=tf.float32)
        dist = tf.abs(
            params["energy_cont_mu"][:, :, None] - self.line_positions_y[None, None, :]
        )
        return tf.reduce_mean(
            tf.reduce_sum(
                tf.square(tf.nn.relu(self.continuum_repulsion_margin - dist)),
                axis=[1, 2],
            )
        )

    def _continuum_balance(self, params):
        """Penalizes uneven batch-mean usage across the n_continuum_components
        sub-components (renormalized among themselves) relative to uniform -
        counters the component collapse confirmed empirically when K>1 with
        no supervision or diversity pressure telling the K slots how to
        divide responsibility (see class docstring). No-op when
        n_continuum_components == 1."""
        if self.n_continuum_components <= 1:
            return tf.constant(0.0, dtype=tf.float32)

        gate_probs = tf.nn.softmax(params["energy_gate_logits"], axis=-1)
        cont_probs = gate_probs[:, :self.n_continuum_components]
        cont_total = tf.reduce_sum(cont_probs, axis=-1, keepdims=True) + 1e-8
        cont_probs_renorm = cont_probs / cont_total
        mean_usage = tf.reduce_mean(cont_probs_renorm, axis=0)

        uniform = 1.0 / float(self.n_continuum_components)
        return tf.reduce_sum(tf.square(mean_usage - uniform))

    def _reconstruction_terms(self, y_cont_true, params):

        ur_true = y_cont_true[:, 0:1]
        uv_true = y_cont_true[:, 1:2]
        phi_r_true = y_cont_true[:, 2:3]
        phi_v_true = y_cont_true[:, 3:4]
        energy_y_true = y_cont_true[:, 4]
        gate_target = y_cont_true[:, 5:5 + self.n_lines + 1]

        if self.continuum_mode == "flow":
            flow_log_prob = self.continuum_flow.log_prob(
                energy_y_true, params["flow_cond"]
            )
            energy_mixture_nll = flow_line_mixture_nll(
                energy_y_true,
                params["energy_gate_logits"],
                flow_log_prob,
                self.line_positions_y,
                self._line_logsigma_clipped(),
            )
        else:
            comp_mu, comp_logsigma = self._mixture_components(params)
            energy_mixture_nll = gaussian_mixture_nll(
                energy_y_true,
                params["energy_gate_logits"],
                comp_mu,
                comp_logsigma,
            )

        # Auxiliary gate supervision (see class docstring): pulls
        # softmax(gate_logits) toward a per-event target built from physical
        # proximity to the known line positions, independent of z - without
        # this the gate can collapse to all-continuum since the encoder can
        # leak line identity into z instead. gate_target only distinguishes
        # {continuum, line_1..line_L} (build_gate_targets has no notion of
        # which continuum sub-component an event belongs to - that split is
        # left unsupervised), so when n_continuum_components > 1 the K
        # continuum slots' probability mass is pooled via log-sum-exp before
        # comparing against gate_target; this reduces to plain log_softmax
        # when n_continuum_components == 1.
        log_norm = tf.reduce_logsumexp(params["energy_gate_logits"], axis=-1, keepdims=True)
        log_cont_prob = tf.reduce_logsumexp(
            params["energy_gate_logits"][:, :self.n_continuum_components],
            axis=-1, keepdims=True,
        ) - log_norm
        log_line_probs = params["energy_gate_logits"][:, self.n_continuum_components:] - log_norm
        log_pooled_gate = tf.concat([log_cont_prob, log_line_probs], axis=-1)

        gate_aux_loss = -tf.reduce_sum(gate_target * log_pooled_gate, axis=-1)

        ur_nll = tf.squeeze(
            gaussian_nll(ur_true, params["ur_mu"], params["ur_logsigma"]),
            axis=1,
        )

        uv_nll = tf.squeeze(
            gaussian_nll(uv_true, params["uv_mu"], params["uv_logsigma"]),
            axis=1,
        )

        phi_r_nll = tf.squeeze(
            gaussian_nll(
                phi_r_true,
                params["phi_r_mu"],
                params["phi_r_logsigma"],
            ),
            axis=1,
        )

        phi_v_nll = tf.squeeze(
            gaussian_nll(
                phi_v_true,
                params["phi_v_mu"],
                params["phi_v_logsigma"],
            ),
            axis=1,
        )

        rec_per = (
            self.task_weights["energy"] * energy_mixture_nll +
            self.w_gate_aux             * gate_aux_loss +
            self.task_weights["ur"]     * ur_nll +
            self.task_weights["uv"]     * uv_nll +
            self.task_weights["phi_r"]  * phi_r_nll +
            self.task_weights["phi_v"]  * phi_v_nll
        )

        pieces = {
            "energy_mixture_nll": energy_mixture_nll,
            "gate_aux_loss": gate_aux_loss,
            "ur_nll": ur_nll,
            "uv_nll": uv_nll,
            "phi_r_nll": phi_r_nll,
            "phi_v_nll": phi_v_nll,
        }

        return rec_per, pieces

    # ==========================================================
    # Keras train/test
    # ==========================================================
    def train_step(self, data):
        (y_cont, E_idx_true, cond), _ = data

        x_in = tf.concat([y_cont, cond], axis=1)

        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(x_in, training=True)
            z = self.sample_z(z_mean, z_logvar)

            params = self._decode_params(z, cond, training=True)

            rec_per, pieces = self._reconstruction_terms(
                y_cont,
                params,
            )

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
                axis=1,
            )
            kl = tf.reduce_mean(kl_per)

            sigma_reg = self._sigma_regularizer(params)
            continuum_repulsion = self._continuum_repulsion(params)
            continuum_balance = self._continuum_balance(params)

            loss = (
                rec + self.beta * kl + sigma_reg
                + self.w_continuum_repulsion * continuum_repulsion
                + self.w_continuum_balance * continuum_balance
            )

        grads = tape.gradient(loss, self.trainable_variables)

        grads_and_vars = []
        seen = set()

        for g, v in zip(grads, self.trainable_variables):
            if g is None:
                continue

            vid = id(v)
            if vid in seen:
                continue

            seen.add(vid)
            grads_and_vars.append((g, v))

        self.optimizer.apply_gradients(grads_and_vars)

        self.loss_tracker.update_state(loss)
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        self.nll_tracker.update_state(nll)

        self.sigma_reg_tracker.update_state(sigma_reg)

        self.energy_mixture_nll_tracker.update_state(tf.reduce_mean(pieces["energy_mixture_nll"]))
        self.gate_aux_loss_tracker.update_state(tf.reduce_mean(pieces["gate_aux_loss"]))
        self.continuum_repulsion_tracker.update_state(continuum_repulsion)
        self.continuum_balance_tracker.update_state(continuum_balance)
        self.ur_nll_tracker.update_state(tf.reduce_mean(pieces["ur_nll"]))
        self.uv_nll_tracker.update_state(tf.reduce_mean(pieces["uv_nll"]))
        self.phi_r_nll_tracker.update_state(tf.reduce_mean(pieces["phi_r_nll"]))
        self.phi_v_nll_tracker.update_state(tf.reduce_mean(pieces["phi_v_nll"]))

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (y_cont, E_idx_true, cond), _ = data

        x_in = tf.concat([y_cont, cond], axis=1)

        z_mean, z_logvar = self.encoder(x_in, training=False)
        z = self.sample_z(z_mean, z_logvar)

        params = self._decode_params(z, cond, training=False)

        rec_per, pieces = self._reconstruction_terms(
            y_cont,
            params,
        )

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
            axis=1,
        )
        kl = tf.reduce_mean(kl_per)

        sigma_reg = self._sigma_regularizer(params)
        continuum_repulsion = self._continuum_repulsion(params)
        continuum_balance = self._continuum_balance(params)

        loss = (
            rec + self.beta * kl + sigma_reg
            + self.w_continuum_repulsion * continuum_repulsion
            + self.w_continuum_balance * continuum_balance
        )

        self.loss_tracker.update_state(loss)
        self.rec_tracker.update_state(rec)
        self.kl_tracker.update_state(kl)
        self.nll_tracker.update_state(nll)

        self.sigma_reg_tracker.update_state(sigma_reg)

        self.energy_mixture_nll_tracker.update_state(tf.reduce_mean(pieces["energy_mixture_nll"]))
        self.gate_aux_loss_tracker.update_state(tf.reduce_mean(pieces["gate_aux_loss"]))
        self.continuum_repulsion_tracker.update_state(continuum_repulsion)
        self.continuum_balance_tracker.update_state(continuum_balance)
        self.ur_nll_tracker.update_state(tf.reduce_mean(pieces["ur_nll"]))
        self.uv_nll_tracker.update_state(tf.reduce_mean(pieces["uv_nll"]))
        self.phi_r_nll_tracker.update_state(tf.reduce_mean(pieces["phi_r_nll"]))
        self.phi_v_nll_tracker.update_state(tf.reduce_mean(pieces["phi_v_nll"]))

        return {m.name: m.result() for m in self.metrics}

    # ==========================================================
    # Sampling / generation
    # ==========================================================
    @tf.function(reduce_retracing=True)
    def decode(self, z, cond):
        return self._decode_params(z, cond, training=False)

    def generate(self, cond, n_samples):
        z = tf.random.normal((n_samples, self.latent_dim))
        params = self.decode(z, cond)

        gate_logits = params["energy_gate_logits"] / self.energy_sampling_temperature
        comp_idx = tf.random.categorical(gate_logits, num_samples=1)
        comp_idx = tf.cast(tf.squeeze(comp_idx, axis=1), tf.int32)

        batch = tf.shape(comp_idx)[0]

        if self.continuum_mode == "flow":
            # Continuum slot (idx 0) is sampled from the flow; line slots
            # (idx 1..n_lines) from their fixed Gaussian. Both are computed
            # for every row and selected by comp_idx.
            flow_sample = self.continuum_flow.sample(params["flow_cond"])

            line_mu_b = tf.broadcast_to(
                self.line_positions_y[None, :], (batch, self.n_lines)
            )
            line_idx = tf.clip_by_value(comp_idx - 1, 0, self.n_lines - 1)
            chosen_line_mu = tf.gather_nd(
                line_mu_b, tf.stack([tf.range(batch), line_idx], axis=1)
            )
            energy_eps = tf.random.normal((batch,))
            line_sample = chosen_line_mu + tf.exp(self._line_logsigma_clipped()) * energy_eps

            energy_y = tf.where(comp_idx == 0, flow_sample, line_sample)
        else:
            comp_mu, comp_logsigma = self._mixture_components(params)
            gather_idx = tf.stack([tf.range(batch), comp_idx], axis=1)
            chosen_mu = tf.gather_nd(comp_mu, gather_idx)
            chosen_logsigma = tf.gather_nd(comp_logsigma, gather_idx)

            energy_eps = tf.random.normal((batch,))
            energy_y = chosen_mu + tf.exp(chosen_logsigma) * energy_eps

        ur_eps = tf.random.normal(tf.shape(params["ur_mu"]))
        ur_q = params["ur_mu"] + tf.exp(params["ur_logsigma"]) * ur_eps

        uv_eps = tf.random.normal(tf.shape(params["uv_mu"]))
        uv_q = params["uv_mu"] + tf.exp(params["uv_logsigma"]) * uv_eps

        phi_r_eps = tf.random.normal(tf.shape(params["phi_r_mu"]))
        phi_r_q = params["phi_r_mu"] + tf.exp(params["phi_r_logsigma"]) * phi_r_eps

        phi_v_eps = tf.random.normal(tf.shape(params["phi_v_mu"]))
        phi_v_q = params["phi_v_mu"] + tf.exp(params["phi_v_logsigma"]) * phi_v_eps

        y_cont = tf.concat(
            [
                ur_q,
                uv_q,
                phi_r_q,
                phi_v_q,
            ],
            axis=1,
        )

        return {
            "energy_y": energy_y,
            "energy_component_idx": comp_idx,
            "y_cont": y_cont,
            "params": params,
        }
