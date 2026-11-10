#!/usr/bin/env python
"""Post-hoc refit of the v0.8 energy head with supervised line routing
(energy_line_routing="supervised", core.losses.flow_line_routed_nll).

Why: on v0_8_2_CR_ingoingfix (and its prior refits) ~1.7% of continuum-routed
gammas land within 50 eV of 511 keV, the generated line comes out ~1.4x too
strong, and the excess carries continuum-zone z and geometry - the dominant
c>0.8 classifier hole. The spike sits in the flow's CDF warp, whose knots were
fitted with the 511 keV atom included (magi.fit_cdf_warp_knots, exclude_y), so
only REFIT_FRESH=warp removes it; routing alone leaves it in place.

Result (REFIT_FRESH=warp REFIT_EPOCHS=12 REFIT_LR=3e-5):
trained_models/v0_8_2_CR_ingoingfix_priorrefit_wide_warpfix.

What is refit: continuum_flow, energy_cont_head (the flow's conditioner) and
energy_gate_head. Everything upstream of energy_feat - encoder, decoder stem,
energy branch, geometry heads, latent prior - stays frozen, so the latent
space and the (already refit) prior remain valid as they are. Each step draws
fresh z ~ q(z|x_train) and minimizes the routed NLL; the route is
argmax(gate_target), the same exact line identification used in training.

Usage:
    python tools/refit_energy_flow_posthoc.py [src_run_dir] [tag]
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_prior_posthoc as rp  # noqa: E402  (sets CPU-only env, seeds)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import magi  # noqa: E402
from magi.core.losses import flow_line_routed_nll  # noqa: E402

ROOT, NAME = rp.ROOT, rp.NAME
SRC = f"{ROOT}/trained_models/v0_8_2_CR_ingoingfix_priorrefit_wide"
BATCH = 4096
EPOCHS = int(os.environ.get("REFIT_EPOCHS", 12))
LR = float(os.environ.get("REFIT_LR", 3e-4))
KEV_Y = 1e-3 / (0.511 * np.log(10.0))  # d(log10 E)/dE at 511 keV, per keV


def energy_params(model, z, cond):
    """Gate logits and flow conditioner, exactly as _decode_params builds them
    for energy_flow_condition="z_cond", energy_condition_geometry=False."""
    stem = model.decoder_stem(tf.concat([z, cond], 1), training=False)
    feat = model.energy_branch(stem, training=False)
    return model.energy_gate_head(feat), model.energy_cont_head(feat, training=True)


def routed_loss(model, zm, zl, cond, y, route):
    z = zm + tf.exp(0.5 * zl) * tf.random.normal(tf.shape(zm))
    gate, fc = energy_params(model, z, cond)
    flp = model.continuum_flow.log_prob(y, fc)
    return tf.reduce_mean(flow_line_routed_nll(
        y, gate, flp, model.line_positions_y, model._line_logsigma_clipped(), route))


def spike_diag(model, zm, zl, yt, gamma_idx, n=200_000, seed=0):
    """Line-window mass the flow itself puts out, on val gammas: fraction of
    continuum-flow samples within 0.05 keV and 2 keV of each line, plus the
    gate's mean line probability."""
    tf.random.set_seed(seed)
    idx = np.flatnonzero(yt == gamma_idx)[:n]
    z = tf.constant(zm[idx]) + tf.exp(0.5 * tf.constant(zl[idx])) * tf.random.normal((len(idx), zm.shape[1]))
    cond = tf.one_hot(yt[idx], model.n_types)
    gate, fc = energy_params(model, z, cond)
    ys = model.continuum_flow.sample(fc).numpy()
    d = np.abs(ys[:, None] - model.line_positions_y.numpy()[None, :]) / KEV_Y
    p = tf.nn.softmax(gate).numpy()
    return {"flow_within_0.05keV": float((d < 0.05).any(1).mean()),
            "flow_within_2keV": float((d < 2.0).any(1).mean()),
            "gate_p_line": float(p[:, 1:].sum(1).mean())}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    tag = sys.argv[2] if len(sys.argv) > 2 else "flowrouted"
    out = f"{src}_{tag}"
    cfg, R, dp, sp = rp.build_dataset()
    n_types, i2t = dp["n_types"], dp["idx_to_type"]
    cfg_src = json.load(open(f"{src}/{NAME}_config.json"))
    model = magi.load_task_adaptive_model_for_generation(
        save_dir=src, model_name=NAME, model_config=cfg_src, n_types=n_types, radius=R, verbose=0)
    gamma = next(k for k, v in i2t.items() if v == "gamma")
    nl = model.n_lines + 1

    def arrays(split):
        X, yt = sp[f"X_cont_{split}"], sp[f"y_{split}"]
        zm, zl, _ = rp.encode(model, X, yt, n_types)
        route = np.eye(nl, dtype=np.float32)[np.argmax(X[:, 5:5 + nl], 1)]
        return zm, zl, yt, X[:, 4].astype(np.float32), route

    cache = os.environ.get("REFIT_CACHE")
    if cache and os.path.exists(cache):
        c = np.load(cache)
        tr, va = (tuple(c[f"tr{i}"] for i in range(5)), tuple(c[f"va{i}"] for i in range(5)))
    else:
        tr, va = arrays("train"), arrays("val")
        if cache:
            np.savez(cache, **{f"tr{i}": a for i, a in enumerate(tr)},
                     **{f"va{i}": a for i, a in enumerate(va)})
    fresh_mode = os.environ.get("REFIT_FRESH", "0")  # "1" re-init flow, "warp" new warp only
    if fresh_mode in ("1", "warp"):
        # Re-initialize the flow and its conditioner. A warm start cannot shed
        # the spike: once line events are routed away, the only pressure on it
        # is the ~1.5% of normalization it wastes, log(1 - 0.015) ~ 0.015 nats,
        # and a fine-tune leaves it in place (measured: 1.55% -> 1.56% of flow
        # samples within 50 eV of 511 keV after 2 epochs).
        #
        # The flow's CDF warp is refit too: the original knots were fitted on
        # all energy_y, 511 keV included, so the warp itself carried the spike
        # (magi.fit_cdf_warp_knots, exclude_y). Refit on the train split with
        # the line positions excluded, same n_knots/eps as the CR notebook.
        yk, zk = magi.fit_cdf_warp_knots(tr[3], n_knots=256, eps=1e-4,
                                         exclude_y=model.line_positions_y.numpy())
        cfg_src = dict(cfg_src, continuum_flow_warp_y_knots=yk.tolist(),
                       continuum_flow_warp_z_knots=zk.tolist())
        cls = type(model); lw = cls.load_weights
        cls.load_weights = lambda self, path: None
        try:
            fresh = magi.load_task_adaptive_model_for_generation(
                save_dir=src, model_name=NAME, model_config=cfg_src, n_types=n_types,
                radius=R, verbose=0)
        finally:
            cls.load_weights = lw
        # By object identity: Keras' auto-numbered variable paths do not carry
        # the owning layer's name for every sublayer.
        ids = lambda m: {id(v) for v in m.continuum_flow.weights + m.energy_cont_head.weights}
        ids_a, ids_b = ids(model), ids(fresh)
        assert len(model.weights) == len(fresh.weights)
        n_re = 0
        for a, b in zip(model.weights, fresh.weights):
            # Same build order; only the name suffixes differ.
            assert a.shape == b.shape and (id(a) in ids_a) == (id(b) in ids_b)
            if id(a) in ids_a and fresh_mode == "1":
                n_re += 1
            else:
                b.assign(a)
        print(f"fresh flow: re-initialized {n_re} variables", flush=True)
        model = fresh
    print(f"train {len(tr[2]):,}  val {len(va[2]):,}  line-routed train "
          f"{tr[4][:, 1:].sum() / len(tr[4]):.4f}", flush=True)

    def val_loss():
        tf.random.set_seed(1)
        tot = 0.0
        for s in range(0, len(va[2]), 50_000):
            b = slice(s, s + 50_000)
            tot += float(routed_loss(model, va[0][b], va[1][b], tf.one_hot(va[2][b], n_types),
                                     va[3][b], va[4][b])) * len(va[2][b])
        return tot / len(va[2])

    train_vars = (model.continuum_flow.trainable_variables
                  + model.energy_cont_head.trainable_variables
                  + model.energy_gate_head.trainable_variables)
    steps = int(np.ceil(len(tr[2]) / BATCH))
    opt = tf.keras.optimizers.Adam(
        tf.keras.optimizers.schedules.CosineDecay(LR, EPOCHS * steps), clipnorm=5.0)

    @tf.function
    def step(zm, zl, yt, y, route):
        with tf.GradientTape() as tape:
            loss = routed_loss(model, zm, zl, tf.one_hot(yt, n_types), y, route)
        opt.apply_gradients(zip(tape.gradient(loss, train_vars), train_vars))
        return loss

    d0, v0 = spike_diag(model, va[0], va[1], va[2], gamma), val_loss()
    print(f"before: val routed NLL {v0:.4f}  {d0}", flush=True)
    hist = {"val_routed_nll": [v0], "spike": [d0]}
    best, best_w = v0, [v.numpy() for v in train_vars]
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        perm = rng.permutation(len(tr[2]))
        tl = 0.0
        for s in range(steps):
            b = perm[s * BATCH:(s + 1) * BATCH]
            tl += float(step(tr[0][b], tr[1][b], tr[2][b], tr[3][b], tr[4][b]))
        v, d = val_loss(), spike_diag(model, va[0], va[1], va[2], gamma)
        hist["val_routed_nll"].append(v); hist["spike"].append(d)
        print(f"epoch {ep + 1:2d}  train {tl / steps:.4f}  val {v:.4f}  {d}", flush=True)
        if v < best:
            best, best_w = v, [w.numpy() for w in train_vars]
    for w, b in zip(train_vars, best_w):
        w.assign(b)

    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if not f.endswith(".weights.h5"):
            shutil.copy2(f"{src}/{f}", f"{out}/{f}")
    model.save_weights(f"{out}/{NAME}.weights.h5")
    for fn, key in ((f"{out}/{NAME}_config.json", None), (f"{out}/{NAME}_metadata.json", "model_config")):
        d = json.load(open(fn))
        tgt = d if key is None else d[key]
        tgt["energy_line_routing"] = "supervised"
        for k in ("continuum_flow_warp_y_knots", "continuum_flow_warp_z_knots"):
            tgt[k] = cfg_src[k]
        if key is not None:
            d["notes"] = (d.get("notes", "") + f" | Post-hoc energy-head refit '{tag}' "
                          f"(tools/refit_energy_flow_posthoc.py): continuum_flow, "
                          f"energy_cont_head, energy_gate_head refit with supervised line "
                          f"routing; everything else frozen from {os.path.basename(src)}. "
                          f"val routed NLL {v0:.4f} -> {best:.4f}.")
        json.dump(d, open(fn, "w"), indent=2)
    json.dump(hist, open(f"{out}/{NAME}_energy_refit_history.json", "w"), indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
