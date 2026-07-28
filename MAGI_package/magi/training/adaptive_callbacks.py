
"""
Adaptive training callbacks for MAGI.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.stats import wasserstein_distance


class TaskAdaptiveLossScheduler(keras.callbacks.Callback):
    """
    Reduce selected task weights when their monitored validation metric:
    1) is below a target threshold
    2) has reached a plateau for a given patience

    `task_configs` maps a task name in `model.task_weights` to a dict with
    `monitor` (a key in the epoch logs) and optionally `threshold`, `patience`,
    `min_delta`, `decay_factor`, `min_weight` and `cooldown`.

    KNOWN LIMITATION: the model's task weights are Python floats read inside
    train_step, which Keras traces into a tf.function once per fit() call.
    Reducing a weight from this callback therefore updates the attribute and
    prints the change, but does NOT alter the compiled graph - the loss keeps
    the weights it was traced with until the next fit(). Verified by zeroing a
    weight mid-fit and observing no discontinuity in the loss. The fix (making
    the weights tf.Variables) is v0.8.2 item 1a in docs/v0.8.2_plan.md; until
    then, treat this callback as reporting-only and set the weights you want
    before training starts.
    """

    def __init__(self, task_configs, verbose=1):
        super().__init__()
        self.task_configs = task_configs
        self.verbose = verbose
        self.state = {}

    def on_train_begin(self, logs=None):
        """Reset the per-task plateau/cooldown bookkeeping."""
        self.state = {}
        for task_name, cfg in self.task_configs.items():
            self.state[task_name] = {
                "best": np.inf,
                "wait": 0,
                "cooldown_counter": 0,
                "n_reductions": 0,
                "last_reduction_epoch": None,
            }

    def on_epoch_end(self, epoch, logs=None):
        """Check each task's monitor and decay its weight if it has plateaued."""
        logs = logs or {}

        for task_name, cfg in self.task_configs.items():
            monitor = cfg["monitor"]
            threshold = float(cfg.get("threshold", np.inf))
            patience = int(cfg.get("patience", 3))
            min_delta = float(cfg.get("min_delta", 1e-4))
            decay_factor = float(cfg.get("decay_factor", 0.5))
            min_weight = float(cfg.get("min_weight", 0.0))
            cooldown = int(cfg.get("cooldown", 0))

            current = logs.get(monitor)
            if current is None:
                continue

            st = self.state[task_name]

            if st["cooldown_counter"] > 0:
                st["cooldown_counter"] -= 1

            if current < (st["best"] - min_delta):
                st["best"] = current
                st["wait"] = 0
            else:
                st["wait"] += 1

            if (current <= threshold) and (st["wait"] >= patience) and (st["cooldown_counter"] == 0):
                if not hasattr(self.model, "task_weights") or task_name not in self.model.task_weights:
                    if self.verbose:
                        print(
                            f"\n[TaskAdaptiveLossScheduler] Skipping task='{task_name}' "
                            f"because it is not present in model.task_weights."
                        )
                    continue
                old_w, new_w = self.model.decay_task_weight(
                    task_name,
                    factor=decay_factor,
                    min_value=min_weight,
                )

                st["wait"] = 0
                st["cooldown_counter"] = cooldown
                st["n_reductions"] += 1
                st["last_reduction_epoch"] = int(epoch + 1)

                if self.verbose:
                    print(
                        f"\n[TaskAdaptiveLossScheduler] Epoch {epoch + 1}: "
                        f"task='{task_name}' monitor='{monitor}'={current:.6f} "
                        f"<= threshold={threshold:.6f} and plateau reached. "
                        f"Reducing weight: {old_w:.6f} -> {new_w:.6f}"
                    )


class TaskAdaptiveTrainingMonitor(keras.callbacks.Callback):
    """
    Compact training monitor for task-adaptive models.

    Prints at the end of each epoch:
    - selected training/validation metrics
    - learning rate
    - current task weights
    - optional scheduler state
    """

    def __init__(
        self,
        scheduler=None,
        metrics_to_show=None,
        show_task_weights=True,
        show_scheduler_state=True,
        every_n_epochs=1,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.metrics_to_show = metrics_to_show or [
            "loss",
            "val_loss",
            "rec",
            "val_rec",
            "kl",
            "val_kl",
            "nll",
            "val_nll",

            # energy, shared
            "energy_ce",
            "val_energy_ce",

            # v0.6 legacy
            "sr_nll",
            "val_sr_nll",
            "uv_ce",
            "val_uv_ce",
            "xy_mse",
            "val_xy_mse",
            "vxy_mse",
            "val_vxy_mse",
            "u_r_mse",
            "val_u_r_mse",

            # v0.7 / v0.7.2 continuous geometry
            "ur_nll",
            "val_ur_nll",
            "uv_nll",
            "val_uv_nll",

            # v0.7 cos/sin phi
            "phi_r_mse",
            "val_phi_r_mse",
            "phi_v_mse",
            "val_phi_v_mse",
            "phi_v_ang",
            "val_phi_v_ang",
            "phi_v_loss",
            "val_phi_v_loss",

            # v0.7.2 quantile phi
            "phi_r_nll",
            "val_phi_r_nll",
            "phi_v_nll",
            "val_phi_v_nll",

            # regularization, when present
            "sigma_reg",
            "val_sigma_reg",
            "phi_reg",
            "val_phi_reg",
            "phi_r_reg",
            "val_phi_r_reg",
        ]
        self.show_task_weights = show_task_weights
        self.show_scheduler_state = show_scheduler_state
        self.every_n_epochs = int(every_n_epochs)

    def _get_lr(self):
        opt = self.model.optimizer
        lr = opt.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            return float(lr(opt.iterations).numpy())
        try:
            return float(tf.keras.backend.get_value(lr))
        except Exception:
            return None

    def on_epoch_end(self, epoch, logs=None):
        """Print the selected metrics, the learning rate and the task weights."""
        logs = logs or {}

        if ((epoch + 1) % self.every_n_epochs) != 0:
            return

        print(f"\n[TrainingMonitor] Epoch {epoch + 1}")

        lr = self._get_lr()
        if lr is not None:
            print(f"  lr = {lr:.6e}")

        # Metrics
        print("  Metrics:")
        for k in self.metrics_to_show:
            if k in logs:
                print(f"    {k:>16s} : {float(logs[k]):.6f}")

        # Task weights
        if self.show_task_weights and hasattr(self.model, "task_weights"):
            print("  Task weights:")
            for k, v in self.model.task_weights.items():
                print(f"    {k:>16s} : {float(v):.6f}")

        # Scheduler state
        if self.show_scheduler_state and (self.scheduler is not None):
            if hasattr(self.scheduler, "state"):
                print("  Adaptive scheduler state:")
                for task_name, st in self.scheduler.state.items():
                    print(
                        f"    {task_name:>16s} : "
                        f"best={float(st['best']):.6f}, "
                        f"wait={int(st['wait'])}, "
                        f"cooldown={int(st['cooldown_counter'])}, "
                        f"reductions={int(st['n_reductions'])}, "
                        f"last_epoch={st['last_reduction_epoch']}"
                    )


class ValidationEnergyDistributionMonitor(keras.callbacks.Callback):
    """
    Compute validation energy distribution metrics at epoch end.

    It adds to logs:
      - val_energy_hist_err
      - val_energy_wdist
    """

    def __init__(
        self,
        E_val_raw,
        energy_bins,
        type_probs,
        n_types,
        n_samples=None,
        energy_mode="uniform",
        min_real_count=5,
        seed=42,
        every_n_epochs=1,
        verbose=1,
    ):
        super().__init__()

        self.E_val_raw = np.asarray(E_val_raw, dtype=np.float64)
        self.energy_bins = np.asarray(energy_bins, dtype=np.float64)
        self.type_probs = np.asarray(type_probs, dtype=np.float64)
        self.type_probs = self.type_probs / self.type_probs.sum()

        self.n_types = int(n_types)
        self.n_samples = n_samples
        self.energy_mode = energy_mode
        self.min_real_count = int(min_real_count)
        self.seed = int(seed)
        self.every_n_epochs = int(every_n_epochs)
        self.verbose = verbose

    def _sample_types(self, n_samples, rng):
        idx = rng.choice(
            len(self.type_probs),
            size=n_samples,
            p=self.type_probs,
        )
        return idx.astype(np.int32)

    def _one_hot_from_idx(self, idx):
        return tf.one_hot(idx, depth=self.n_types, dtype=tf.float32)

    def _energy_from_idx(self, idx, rng):
        idx = np.asarray(idx, dtype=np.int32)
        left = self.energy_bins[idx]
        right = self.energy_bins[idx + 1]

        if self.energy_mode == "uniform":
            u = rng.uniform(0.0, 1.0, size=len(idx))
            return left + u * (right - left)

        if self.energy_mode == "bin_center":
            return 0.5 * (left + right)

        raise ValueError("energy_mode must be 'uniform' or 'bin_center'")

    def _hist_quality_metrics(self, real, gen):
        real_counts, edges = np.histogram(real, bins=self.energy_bins)
        gen_counts, _ = np.histogram(gen, bins=edges)

        real_prob = real_counts.astype(np.float64)
        gen_prob = gen_counts.astype(np.float64)

        real_prob /= max(real_prob.sum(), 1.0)
        gen_prob /= max(gen_prob.sum(), 1.0)

        mask = real_counts >= self.min_real_count

        if not np.any(mask):
            return {
                "mean_rel_err": np.inf,
                "max_rel_err": np.inf,
                "p90_rel_err": np.inf,
                "n_valid_bins": 0,
                "valid_bin_fraction": 0.0,
                "quality_score": np.inf,
            }

        rel_err = np.abs(gen_prob[mask] / np.maximum(real_prob[mask], 1e-12) - 1.0)

        mean_rel_err = float(np.mean(rel_err))
        max_rel_err = float(np.max(rel_err))
        p90_rel_err = float(np.percentile(rel_err, 90))

        quality_score = max(
            mean_rel_err / 0.10,
            p90_rel_err / 0.15,
            max_rel_err / 0.25,
        )

        return {
            "mean_rel_err": mean_rel_err,
            "max_rel_err": max_rel_err,
            "p90_rel_err": p90_rel_err,
            "n_valid_bins": int(np.sum(mask)),
            "valid_bin_fraction": float(np.mean(mask)),
            "quality_score": float(quality_score),
        }

    def on_epoch_end(self, epoch, logs=None):
        """Generate a validation-sized sample and score its energy spectrum.

        Adds the val_energy_* keys to `logs`, so they can be monitored by
        EarlyStopping/ReduceLROnPlateau like any other metric.
        """
        logs = logs or {}

        if ((epoch + 1) % self.every_n_epochs) != 0:
            return

        rng = np.random.default_rng(self.seed + epoch)

        n_gen = len(self.E_val_raw)
        if self.n_samples is not None:
            n_gen = min(int(self.n_samples), n_gen)

        real_sample = self.E_val_raw
        if len(real_sample) > n_gen:
            idx = rng.choice(len(real_sample), size=n_gen, replace=False)
            real_sample = real_sample[idx]

        gen_type_idx = self._sample_types(n_gen, rng)
        gen_cond = self._one_hot_from_idx(gen_type_idx)

        gen_out = self.model.generate(gen_cond, n_gen)
        energy_idx_gen = gen_out["energy_idx"].numpy().astype(np.int32)

        E_gen = self._energy_from_idx(energy_idx_gen, rng)

        q = self._hist_quality_metrics(real_sample, E_gen)

        mean_rel_err = q["mean_rel_err"]
        max_rel_err = q["max_rel_err"]
        p90_rel_err = q["p90_rel_err"]
        quality_score = q["quality_score"]

        log_real = np.log10(real_sample)
        log_gen = np.log10(E_gen)

        wdist = float(
            wasserstein_distance(log_real, log_gen) / np.maximum(np.std(log_real), 1e-12)
        )

        logs["val_energy_mean_rel_err"] = mean_rel_err
        logs["val_energy_max_rel_err"] = max_rel_err
        logs["val_energy_p90_rel_err"] = p90_rel_err
        logs["val_energy_quality_score"] = quality_score
        logs["val_energy_n_valid_bins"] = q["n_valid_bins"]
        logs["val_energy_valid_bin_fraction"] = q["valid_bin_fraction"]
        logs["val_energy_hist_err"] = mean_rel_err  # backward compatibility
        logs["val_energy_wdist"] = wdist

        if self.verbose:
            print(
                f"\n[ValidationEnergyDistributionMonitor] Epoch {epoch + 1}: "
                f"mean_rel_err={mean_rel_err:.6f}, "
                f"max_rel_err={max_rel_err:.6f}, "
                f"p90_rel_err={p90_rel_err:.6f}, "
                f"quality_score={quality_score:.6f}, "
                f"valid_bins={q['n_valid_bins']}, "
                f"val_energy_wdist={wdist:.6f}"
            )