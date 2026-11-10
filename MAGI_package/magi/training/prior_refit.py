"""
Prior refit for MAGI: the last stage of the training recipe.

The learned prior p(z | cond) of the v0.8 models is trained jointly with the
encoder and decoder, through a Monte-Carlo KL term that shares every step with
the reconstruction loss. It does not reach the aggregated posterior it is meant
to match, and the mismatch is the dominant part of the classifier AUC excess of
the generated samples (docs/CVAE_vs_AutoRegFlow.md, Phase 0 and A2). Refitting
only the prior on the frozen encoder removes most of it: on CryoSphere-CR the
AUC of a retrained model goes from 0.649 to 0.624 (mean of three training
seeds, standard deviation 0.004 between seeds).

`refit_prior` does the refit on a trained model that is in memory. With the
encoder frozen, z is drawn from q(z | x) over the training split with fresh
noise at every epoch, and a new `ConditionalCouplingPrior` is fitted by maximum
likelihood on (z, prior conditioning). This minimises E_x KL(q(z|x) || p(z|c))
exactly, so the validation KL is the loss, and the weights of the epoch with
the lowest validation KL are kept. `save_refit_model` writes the result as a
normal run directory, and `refit_prior_run` does both for a saved run.

The refit checkpoints itself after every epoch when `checkpoint_dir` is given,
and the same call continues an interrupted fit.

Typical use after `fit_model` and `save_final_trained_model`:

    X_tr, c_tr = split["X_cont_train"], one_hot_cond_train
    result = magi.refit_prior(model, X_tr, c_tr, X_va, c_va,
                              checkpoint_dir="trained_models/my_run/refit")
    magi.save_refit_model(model, result.prior, src_dir="trained_models/my_run",
                          out_dir="trained_models/my_run_final", model_name="mix_CR",
                          n_types=4, radius=R, hidden=result.hidden,
                          n_layers=result.n_layers,
                          val_kl_before=result.val_kl_before, val_kl_after=result.val_kl_after)
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from ..core.priors import ConditionalCouplingPrior

BATCH_SIZE = 4096


@dataclass
class PriorRefitResult:
    """Outcome of `refit_prior`."""

    prior: ConditionalCouplingPrior
    history: list
    val_kl_before: float
    val_kl_after: float
    hidden: tuple
    n_layers: int
    epochs: int
    lr: float
    extra: dict = field(default_factory=dict)


def encode_posterior(model, X, cond, batch_size=100_000):
    """
    Posterior moments of a trained model and its training-time prior conditioning.

    Parameters
    ----------
    model : CVAE_MixEnergy_ContPhi_TaskAdaptive
        A trained model with a learned prior.

    X : np.ndarray, shape (n, n_cont)
        Continuous feature block, as passed to the encoder.

    cond : np.ndarray, shape (n, cond_dim)
        Conditioning vectors, as passed to the encoder.

    batch_size : int
        Events per forward pass.

    Returns
    -------
    tuple of np.ndarray
        `(z_mean, z_logvar, prior_cond)`, where `prior_cond` is the conditioning
        the model builds for its prior during training (the conditioning plus
        the real zone when the prior is zone-conditioned).
    """
    zm, zl, pc = [], [], []
    for s in range(0, len(X), batch_size):
        x = tf.constant(np.asarray(X[s:s + batch_size], dtype=np.float32))
        c = tf.constant(np.asarray(cond[s:s + batch_size], dtype=np.float32))
        m, l = model.encoder(tf.concat([x, c], 1), training=False)
        zm.append(m.numpy())
        zl.append(l.numpy())
        pc.append(model._prior_cond_from_real(c, x).numpy())
    return np.vstack(zm), np.vstack(zl), np.vstack(pc)


def posterior_kl(prior, z_mean, z_logvar, prior_cond, seed=0):
    """
    Monte-Carlo estimate of KL(q(z|x) || p(z|c)) per event, one z per event.

    This is the model's own `val_kl` term, evaluated for any prior.

    Parameters
    ----------
    prior : ConditionalCouplingPrior
        The prior to score.

    z_mean, z_logvar, prior_cond : np.ndarray
        As returned by `encode_posterior`.

    seed : int
        Seed of the noise draw.

    Returns
    -------
    np.ndarray, shape (n,)
        KL estimate of every event, in nats.
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in range(0, len(z_mean), 50_000):
        m, l = z_mean[s:s + 50_000], z_logvar[s:s + 50_000]
        eps = rng.standard_normal(m.shape).astype(np.float32)
        z = m + np.exp(0.5 * l) * eps
        log_q = np.sum(-0.5 * (eps ** 2 + np.log(2 * np.pi)) - 0.5 * l, axis=1)
        log_p = prior.log_prob(tf.constant(z), tf.constant(prior_cond[s:s + 50_000])).numpy()
        out.append(log_q - log_p)
    return np.concatenate(out)


def prior_posterior_auc(model, prior, z_mean, z_logvar, cond, n=200_000, seed=1):
    """
    Classifier AUC separating z ~ q(z|x) from z ~ p(z|cond), pooled and per type.

    0.5 means that the prior reproduces the aggregated posterior. The prior
    conditioning is built as `generate()` builds it, with the zone sampled from
    `zone_probs` when the prior is zone-conditioned.

    Parameters
    ----------
    model : CVAE_MixEnergy_ContPhi_TaskAdaptive
        Provides `n_types`, `prior_zone_conditioning` and `zone_probs`.

    prior : ConditionalCouplingPrior
        The prior to test.

    z_mean, z_logvar : np.ndarray
        Posterior moments of held-out events, from `encode_posterior`.

    cond : np.ndarray, shape (n, cond_dim)
        Conditioning of the same events.

    n : int
        Maximum number of events used.

    seed : int
        Seed of the subsample and of the draws.

    Returns
    -------
    dict
        `{"all": auc, <type index>: auc, ...}`; types with fewer than 2000
        events are left out.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z_mean), size=min(n, len(z_mean)), replace=False)
    zq = z_mean[idx] + np.exp(0.5 * z_logvar[idx]) * rng.standard_normal(z_mean[idx].shape)
    c = tf.constant(np.asarray(cond[idx], dtype=np.float32))
    tf.random.set_seed(seed)
    pc = c
    if model.prior_zone_conditioning:
        zp = model._zone_probs_rows(c)
        zi = tf.squeeze(tf.random.categorical(tf.math.log(zp + 1e-12), 1), 1)
        pc = tf.concat([c, tf.one_hot(zi, model.n_lines + 1)], 1)
    zp_s = prior.sample(pc).numpy()
    types = np.argmax(np.asarray(cond[idx])[:, :model.n_types], axis=1)
    oh = np.eye(model.n_types)[types]

    def auc(a, b):
        X = np.vstack([a, b])
        y = np.r_[np.ones(len(a)), np.zeros(len(b))]
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=63, random_state=0)
        clf.fit(Xa, ya)
        return roc_auc_score(yb, clf.predict_proba(Xb)[:, 1])

    res = {"all": auc(np.hstack([zq, oh]), np.hstack([zp_s, oh]))}
    for k in range(model.n_types):
        sel = types == k
        if sel.sum() > 2000:
            res[k] = auc(zq[sel], zp_s[sel])
    return res


def refit_prior(
    model,
    X_train,
    cond_train,
    X_val,
    cond_val,
    hidden=(128, 128),
    n_layers=6,
    epochs=60,
    lr=1e-3,
    batch_size=BATCH_SIZE,
    checkpoint_dir=None,
    verbose=1,
    sample_weight_train=None,
    sample_weight_val=None,
):
    """
    Refit the learned prior of a trained model on its frozen encoder.

    The defaults are the recipe measured on CryoSphere-CR (a conditioner twice
    as wide as the joint one, a fresh identity initialisation, 60 epochs at
    1e-3 with cosine decay to 5%). The model itself is not modified; use
    `save_refit_model` to write the new prior into a run directory.

    Parameters
    ----------
    model : CVAE_MixEnergy_ContPhi_TaskAdaptive
        A trained model (weights loaded) with `prior="coupling"`.

    X_train, X_val : np.ndarray, shape (n, n_cont)
        Continuous feature blocks of the training and validation splits.

    cond_train, cond_val : np.ndarray, shape (n, cond_dim)
        Conditioning vectors of the same events.

    hidden : tuple of int
        Width of the coupling conditioner.

    n_layers : int
        Number of coupling layers.

    epochs : int
        Epochs of the fit; the cosine decay is set by this value.

    lr : float
        Initial learning rate.

    batch_size : int
        Events per step.

    checkpoint_dir : str or None
        If set, a per-epoch checkpoint is written there and the same call
        continues from it after an interruption.

    verbose : int
        0 silent, 1 one line per epoch.

    sample_weight_train, sample_weight_val : np.ndarray of shape (n,) or None
        Per-event weights of the maximum-likelihood fit and of the validation KL
        that selects the best epoch. A prior that is right on average can still be
        wrong in a sparse region of the conditioning (on DM1.2 the muon primaries
        below 100 MeV, about 10 % of the muons, where the prior over-produces slow
        muons); weighting that region raises its share of the loss. With weights
        the reported `val_kl_after` is the weighted one, and the unweighted value
        of every epoch is kept in the history (`val_kl_unweighted`).

    Returns
    -------
    PriorRefitResult
        The refitted prior (best validation KL), the history, the validation
        KL before and after, and the settings used.
    """
    if not hasattr(model, "prior") or not isinstance(model.prior, ConditionalCouplingPrior):
        raise ValueError("refit_prior needs a model with prior='coupling'")

    tr = encode_posterior(model, X_train, cond_train)
    va = encode_posterior(model, X_val, cond_val)
    latent_dim, cond_dim = tr[0].shape[1], tr[2].shape[1]
    wtr = (np.ones(len(tr[0]), np.float32) if sample_weight_train is None
           else np.asarray(sample_weight_train, np.float32).reshape(-1))
    wva = (np.ones(len(va[0]), np.float32) if sample_weight_val is None
           else np.asarray(sample_weight_val, np.float32).reshape(-1))
    if len(wtr) != len(tr[0]) or len(wva) != len(va[0]):
        raise ValueError("sample weights must have one value per event")

    def kl_of(prior_):
        k = posterior_kl(prior_, *va)
        return float(np.sum(wva * k) / np.sum(wva)), float(k.mean())

    kl_before = kl_of(model.prior)[0]
    spec = {"hidden": list(hidden), "n_layers": int(n_layers), "epochs": int(epochs),
            "lr": float(lr), "batch_size": int(batch_size),
            "weighted": sample_weight_train is not None,
            "weight_sum": round(float(wtr.sum()), 3)}

    prior = ConditionalCouplingPrior(
        latent_dim=latent_dim, cond_dim=cond_dim, n_layers=n_layers, hidden=tuple(hidden),
        log_scale_clamp=model.prior_log_scale_clamp, name="latent_prior")
    prior.log_prob(tf.zeros((1, latent_dim)), tf.zeros((1, cond_dim)))  # create the variables

    zm, zl, pc = (tf.constant(a) for a in tr)
    wt = tf.constant(wtr)
    n = int(zm.shape[0])
    steps = max(n // batch_size, 1)
    sched = tf.keras.optimizers.schedules.CosineDecay(lr, epochs * steps, alpha=0.05)
    opt = tf.keras.optimizers.Adam(sched, clipnorm=5.0)

    @tf.function
    def step(m, l, c, w):
        z = m + tf.exp(0.5 * l) * tf.random.normal(tf.shape(m))
        with tf.GradientTape() as tape:
            loss = -tf.reduce_sum(w * prior.log_prob(z, c)) / tf.reduce_sum(w)
        g = tape.gradient(loss, prior.trainable_variables)
        opt.apply_gradients(zip(g, prior.trainable_variables))
        return loss

    best, best_w, hist, start = np.inf, prior.get_weights(), [], 0
    opt.build(prior.trainable_variables)
    if checkpoint_dir:
        state_f = os.path.join(checkpoint_dir, "state.json")
        arr_f = os.path.join(checkpoint_dir, "arrays.npz")
        if os.path.exists(state_f):
            st = json.load(open(state_f))
            if st["spec"] != spec:
                raise ValueError(f"{checkpoint_dir} was written for {st['spec']}, not {spec}: "
                                 "remove it or use the same settings")
            arr = np.load(arr_f)
            prior.set_weights([arr[f"w{i}"] for i in range(len(prior.get_weights()))])
            for i, v in enumerate(opt.variables):
                v.assign(arr[f"o{i}"])
            best, hist, start = st["best"], st["hist"], st["epochs_done"]
            best_w = [arr[f"b{i}"] for i in range(len(best_w))]
            if verbose:
                print(f"[MAGI] prior refit resumed from {checkpoint_dir}: {start} epochs done, "
                      f"best val KL {best:.4f}", flush=True)

    for ep in range(start, epochs):
        t0 = time.time()
        perm = tf.random.shuffle(tf.range(n))
        tot = 0.0
        for s in range(steps):
            b = perm[s * batch_size:(s + 1) * batch_size]
            tot += float(step(tf.gather(zm, b), tf.gather(zl, b), tf.gather(pc, b), tf.gather(wt, b)))
        kl, kl_u = kl_of(prior)
        hist.append({"epoch": ep + 1, "train_nll": tot / steps, "val_kl": kl, "val_kl_unweighted": kl_u})
        if kl < best:
            best, best_w = kl, prior.get_weights()
        if verbose:
            print(f"[MAGI] prior refit ep {ep + 1:3d}  train -log p {tot / steps:8.4f}  "
                  f"val KL {kl:8.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if checkpoint_dir:
            _save_state(checkpoint_dir, prior, opt, best_w, best, hist, ep + 1, spec)

    prior.set_weights(best_w)
    return PriorRefitResult(
        prior=prior, history=hist, val_kl_before=kl_before, val_kl_after=float(best),
        hidden=tuple(hidden), n_layers=int(n_layers), epochs=int(epochs), lr=float(lr))


def _save_state(directory, prior, opt, best_w, best, hist, epochs_done, spec):
    """Write the refit checkpoint through temporary files, so that a kill in the
    middle of a write leaves the previous epoch's checkpoint intact."""
    os.makedirs(directory, exist_ok=True)
    arrays = {f"w{i}": w for i, w in enumerate(prior.get_weights())}
    arrays.update({f"b{i}": w for i, w in enumerate(best_w)})
    arrays.update({f"o{i}": v.numpy() for i, v in enumerate(opt.variables)})
    np.savez(os.path.join(directory, "arrays_new.npz"), **arrays)
    with open(os.path.join(directory, "state_new.json"), "w") as f:
        json.dump({"spec": spec, "epochs_done": epochs_done, "best": float(best), "hist": hist}, f)
    os.replace(os.path.join(directory, "arrays_new.npz"), os.path.join(directory, "arrays.npz"))
    os.replace(os.path.join(directory, "state_new.json"), os.path.join(directory, "state.json"))


def save_refit_model(
    model,
    prior,
    src_dir,
    out_dir,
    model_name,
    n_types,
    radius,
    hidden,
    n_layers,
    val_kl_before=None,
    val_kl_after=None,
    history=None,
):
    """
    Write a run directory identical to `src_dir` except for the prior.

    Every file of the source run is copied, the weights file is rebuilt with the
    new prior, and the architecture entries (`prior_hidden`, `prior_n_layers`)
    are updated in the config and metadata JSON. A note records the refit.
    Assigning `model.prior = prior` does not work, since Keras keeps the old
    layer's slot in its save order; the model is rebuilt from the updated config
    through the normal loader instead and every non-prior variable is copied in
    order.

    Parameters
    ----------
    model : CVAE_MixEnergy_ContPhi_TaskAdaptive
        The trained model that owns the frozen encoder and decoder.

    prior : ConditionalCouplingPrior
        The refitted prior, from `refit_prior`.

    src_dir : str
        Run directory the model was saved to.

    out_dir : str
        Directory to write; created if needed.

    model_name : str
        Filename stem of the run.

    n_types : int
        Number of particle types.

    radius : float
        Sphere radius in mm, as used in training.

    hidden : tuple of int
        Conditioner width of the new prior.

    n_layers : int
        Number of coupling layers of the new prior.

    val_kl_before, val_kl_after : float or None
        Validation KL before and after, recorded in the note.

    history : list or None
        Refit history, saved as `<model_name>_prior_refit_history.json`.

    Returns
    -------
    str
        `out_dir`.
    """
    from .checkpointing import load_task_adaptive_model_for_generation

    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if not f.endswith(".weights.h5") and os.path.isfile(os.path.join(src_dir, f)):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(out_dir, f))
    cfg = json.load(open(os.path.join(src_dir, f"{model_name}_config.json")))
    cfg2 = dict(cfg, prior_hidden=list(hidden), prior_n_layers=int(n_layers))

    cls = type(model)
    original_load = cls.load_weights
    cls.load_weights = lambda self, path: None  # the new architecture has no weights file yet
    try:
        m2 = load_task_adaptive_model_for_generation(
            save_dir=src_dir, model_name=model_name, model_config=cfg2,
            n_types=n_types, radius=radius, verbose=0)
    finally:
        cls.load_weights = original_load
    m2.prior.log_prob(tf.zeros((1, prior.latent_dim)), tf.zeros((1, prior.cond_dim)))

    is_prior = lambda v: "latent_prior" in v.path
    src = [v for v in model.weights if not is_prior(v)]
    dst = [v for v in m2.weights if not is_prior(v)]
    if len(src) != len(dst) or any(a.shape != b.shape for a, b in zip(src, dst)):
        raise ValueError("the rebuilt model does not match the trained one outside the prior")
    for a, b in zip(src, dst):
        b.assign(a)
    m2.prior.set_weights(prior.get_weights())
    m2.save_weights(os.path.join(out_dir, f"{model_name}.weights.h5"))

    note = (f"Prior refit (magi.training.prior_refit): encoder and decoder frozen from "
            f"{os.path.basename(os.path.normpath(src_dir))}, prior refit by maximum likelihood "
            f"on z ~ q(z|x_train)")
    if val_kl_before is not None and val_kl_after is not None:
        note += f", val KL {val_kl_before:.3f} -> {val_kl_after:.3f}"
    for fn, key in ((f"{model_name}_config.json", None), (f"{model_name}_metadata.json", "model_config")):
        path = os.path.join(out_dir, fn)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        target = d if key is None else d[key]
        target["prior_hidden"] = list(hidden)
        target["prior_n_layers"] = int(n_layers)
        if key is not None:
            d["notes"] = (d.get("notes", "") + " | " + note + ".").strip(" |")
            d["prior_refit"] = {"val_kl_before": val_kl_before, "val_kl_after": val_kl_after,
                                "hidden": list(hidden), "n_layers": int(n_layers)}
        json.dump(d, open(path, "w"), indent=2)
    if history is not None:
        json.dump(history, open(os.path.join(out_dir, f"{model_name}_prior_refit_history.json"), "w"),
                  indent=2)
    return out_dir


def refit_prior_run(
    src_dir,
    out_dir,
    model_name,
    X_train,
    cond_train,
    X_val,
    cond_val,
    n_types,
    radius,
    checkpoint_dir=None,
    **refit_kwargs,
):
    """
    Refit the prior of a saved run and write the result as a new run directory.

    Loads the run, calls `refit_prior` and then `save_refit_model`.

    Parameters
    ----------
    src_dir : str
        Saved run directory (weights, config, metadata, transformers).

    out_dir : str
        Directory of the refitted run.

    model_name : str
        Filename stem of the run.

    X_train, cond_train, X_val, cond_val : np.ndarray
        Continuous features and conditioning of the training and validation
        splits, as in `refit_prior`.

    n_types : int
        Number of particle types.

    radius : float
        Sphere radius in mm, as used in training.

    checkpoint_dir : str or None
        Per-epoch checkpoint of the refit, as in `refit_prior`.

    **refit_kwargs
        Passed to `refit_prior` (hidden, n_layers, epochs, lr, batch_size, verbose).

    Returns
    -------
    PriorRefitResult
        With `extra["out_dir"]` set to the written run directory.
    """
    from .checkpointing import load_task_adaptive_model_for_generation

    cfg = json.load(open(os.path.join(src_dir, f"{model_name}_config.json")))
    model = load_task_adaptive_model_for_generation(
        save_dir=src_dir, model_name=model_name, model_config=cfg,
        n_types=n_types, radius=radius, verbose=0)
    result = refit_prior(model, X_train, cond_train, X_val, cond_val,
                         checkpoint_dir=checkpoint_dir, **refit_kwargs)
    result.extra["out_dir"] = save_refit_model(
        model, result.prior, src_dir, out_dir, model_name, n_types, radius,
        result.hidden, result.n_layers, result.val_kl_before, result.val_kl_after,
        history=result.history)
    return result
