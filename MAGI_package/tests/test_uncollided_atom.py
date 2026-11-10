"""Uncollided-primary atom of CVAE_MixEnergy_ContPhi_TaskAdaptive.

The atom is one extra gate slot whose position is the primary energy of each
event, y = log10(E_prim), recovered from the last conditioning column. The tests
cover what is easy to get wrong: the position inversion, the masking of the slot
for crossings that are not of the primary's type, the exact energy of an
atom-routed generated event, the per-event line positions in the loss, the
config round trip, and the data helpers.
"""
import json

import numpy as np
import pytest
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer

import magi

N_TYPES = 2                      # crossing types: 0 = gamma, 1 = mu-
P2C = [0, 1]                     # primary type 0 -> crossing 0, 1 -> crossing 1
N_PRIM = len(P2C)
COND_DIM = N_TYPES + N_PRIM + 1
LINES = np.array([np.log10(0.5)], dtype=np.float32)
N_LINES = 2                      # one fixed line + the atom
N_ZONES = N_LINES + 1
Y_DIM = 4 + 1 + N_ZONES

_rng = np.random.default_rng(0)
_ey = _rng.uniform(-1.0, 3.0, 5000)            # log10 E_prim
_QT = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(_ey.reshape(-1, 1))


def _model(**kw):
    knots = magi.primary_energy_knots_from_transformer(_QT)
    m = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        gate_focal_gamma=0.0, n_types=N_TYPES, cond_dim=COND_DIM, line_positions_y=LINES,
        latent_dim=4, hidden=(16, 16), beta=0.2, continuum_mode="flow",
        continuum_flow_bins=8, continuum_flow_transforms=2, continuum_flow_warp="affine",
        energy_flow_condition="z_cond", energy_line_routing="supervised", prior="coupling",
        prior_n_layers=2, prior_hidden=(8, 8),
        line_logsigma_init=np.array([-9.0], dtype=np.float32), line_logsigma_trainable=False,
        uncollided_atom=True, primary_energy_knots=knots, primary_to_crossing_type=P2C, **kw)
    c = tf.zeros((2, COND_DIM))
    m.encoder(tf.concat([tf.zeros((2, Y_DIM)), c], 1), training=False)
    m.decode(tf.zeros((2, m.latent_dim)), c)
    return m


def _cond(types, ptypes, ey):
    c = np.zeros((len(types), COND_DIM), np.float32)
    c[np.arange(len(types)), types] = 1
    c[np.arange(len(types)), N_TYPES + np.asarray(ptypes)] = 1
    c[:, -1] = _QT.transform(np.asarray(ey).reshape(-1, 1)).reshape(-1)
    return tf.constant(c)


def test_the_atom_adds_one_slot_and_one_logsigma():
    m = _model()
    assert m.n_fixed_lines == 1 and m.n_lines == 2 and m.y_cont_dim == Y_DIM
    assert m.line_logsigma.shape == (2,)
    assert float(m.line_logsigma[1]) == pytest.approx(np.log(1e-4), rel=1e-5)
    params = m.decode(tf.zeros((1, m.latent_dim)), _cond([0], [0], [0.0]))
    assert params["energy_gate_logits"].shape[-1] == N_ZONES


def test_position_is_log10_primary_energy():
    m = _model()
    ey = np.array([-0.7, 0.3, 1.9, 2.8])
    pos = m._primary_y(_cond([0, 0, 1, 1], [0, 0, 1, 1], ey)).numpy()
    np.testing.assert_allclose(pos, ey, atol=1e-4)  # far below the atom width
    lp = m._line_positions(_cond([0, 1], [0, 1], [0.0, 1.0])).numpy()
    assert lp.shape == (2, 2) and lp[0, 0] == pytest.approx(LINES[0])


def test_atom_is_masked_for_a_crossing_of_another_type():
    m = _model()
    cond = _cond([0, 1, 0, 1], [0, 1, 1, 0], [0.0] * 4)   # own, own, other, other
    z = tf.zeros((4, m.latent_dim))
    logits = m.decode(z, cond)["energy_gate_logits"].numpy()
    assert (logits[:2, -1] > -1e3).all() and (logits[2:, -1] < -1e3).all()


def test_loss_uses_the_event_primary_energy():
    m = _model()
    n = 6
    ey = np.linspace(-0.5, 2.5, n)
    cond = _cond([0] * n, [0] * n, ey)
    gate = np.zeros((n, N_ZONES), np.float32)
    gate[:, -1] = 1.0                                   # all uncollided
    geom = np.zeros((n, 4), np.float32)
    atom_y = m._primary_y(cond).numpy()

    def nll(y_energy):
        y = tf.constant(np.hstack([geom, y_energy[:, None], gate]).astype("float32"))
        z = tf.zeros((n, m.latent_dim))
        _, pieces = m._reconstruction_terms(y, m.decode(z, cond))
        return pieces["energy_mixture_nll"].numpy()

    at, off = nll(atom_y), nll(atom_y + 0.05)           # 500 sigma away
    assert np.isfinite(at).all() and (off > at + 100).all()


def test_generated_atom_events_carry_exactly_the_primary_energy():
    m = _model()
    n = 500
    cond = _cond([0] * n, [0] * n, np.full(n, 1.2))
    # force the gate to the atom: a large positive bias on its output slot
    head = m.energy_gate_head
    out_layer = head.layers[-1] if hasattr(head, "layers") else head
    b = out_layer.bias.numpy().copy()
    b[-1] += 50.0
    out_layer.bias.assign(b)
    out = m.generate(cond, n)
    assert (out["energy_component_idx"].numpy() == m.n_lines).all()
    np.testing.assert_allclose(out["energy_y"].numpy(), m._primary_y(cond).numpy(), atol=1e-6)


def test_config_round_trip_keeps_the_atom():
    cfg = _model().to_generation_config()
    assert cfg["uncollided_atom"] is True and len(cfg["line_positions_y"]) == 1
    assert cfg["primary_to_crossing_type"] == P2C
    assert set(cfg["primary_energy_knots"]) == {"u", "y"}
    json.dumps(cfg)  # serialisable


def test_uncollided_atom_needs_flow_mode_and_matching_cond():
    with pytest.raises(ValueError, match="flow"):
        magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
            n_types=N_TYPES, cond_dim=COND_DIM, line_positions_y=LINES, continuum_mode="gaussian",
            uncollided_atom=True, primary_energy_knots=([0, 1], [0, 1]), primary_to_crossing_type=P2C)
    with pytest.raises(ValueError, match="cond_dim"):
        magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
            n_types=N_TYPES, cond_dim=N_TYPES + 1, line_positions_y=LINES, continuum_mode="flow",
            continuum_flow_warp="affine", uncollided_atom=True,
            primary_energy_knots=([0, 1], [0, 1]), primary_to_crossing_type=P2C)


def test_add_uncollided_gate_target_moves_only_matching_events():
    gt = np.array([[1, 0], [0, 1], [1, 0], [1, 0]], np.float32)
    E = np.array([1.0, 0.5, 1.0, 2.0])
    Ep = np.array([1.0, 0.5, 3.0, 2.0])
    ct = np.array([0, 0, 0, 1])            # last: a mu- crossing of a gamma primary
    pt = np.array([0, 0, 0, 0])
    ext, unc = magi.add_uncollided_gate_target(gt, E, Ep, ct, pt, P2C)
    assert unc.tolist() == [True, True, False, False]
    assert ext.shape == (4, 3) and ext[:2, :-1].sum() == 0 and ext[:2, -1].tolist() == [1, 1]
    np.testing.assert_array_equal(ext[2:, :-1], gt[2:])
    assert ext.sum(axis=1).tolist() == [1, 1, 1, 1]


def test_position_inversion_is_exact_across_the_whole_range():
    m = _model()
    y_true = np.interp(np.random.default_rng(1).uniform(0.001, 0.999, 5000),
                       _QT.references_, _QT.quantiles_[:, 0])
    cond = _cond([0] * 5000, [0] * 5000, y_true)
    err = np.abs(m._primary_y(cond).numpy() - y_true)
    assert np.median(err) < 1e-6 and err.max() < 1e-4      # the atom is 1e-4 wide


def test_zone_table_by_primary_gives_each_group_its_own_fraction():
    rng = np.random.default_rng(5)
    n = 20_000
    t = np.zeros(n, int)                        # all gamma crossings
    p = rng.integers(0, 2, n)                   # primary 0 = gamma, 1 = mu
    q = rng.normal(size=n)
    zone = np.zeros((n, 3), np.float32)
    atom = (p == 0) & (rng.random(n) < 0.55)    # only gamma primaries have uncollided crossings
    zone[atom, 2] = 1
    zone[~atom, 0] = 1
    table, edges = magi.build_zone_probs_table(zone, t, p, q, n_types=N_TYPES, n_primary_types=N_PRIM,
                                               n_energy_bins=4)
    t3 = table.reshape(N_TYPES, N_PRIM, 4, 3)
    assert t3[0, 0, :, 2].min() > 0.45 and t3[0, 1, :, 2].max() == 0.0
    assert table.shape == (N_TYPES * N_PRIM * 4, 3) and edges.shape == (3,)


def test_the_model_draws_the_zone_from_the_group_of_the_event():
    table = np.zeros((N_TYPES * N_PRIM * 2, N_ZONES), np.float32)
    table[:, 0] = 1.0
    table[(0 * N_PRIM + 0) * 2 + 1] = [0.0, 0.0, 1.0]     # gamma crossing of a gamma primary, upper energy bin
    m = _model(prior_zone_conditioning=True, zone_probs=table, zone_primary_energy_edges=[0.0])
    cond = _cond([0, 0, 0, 1], [0, 0, 1, 1], [-0.5, 2.0, 2.0, 2.0])
    rows = m._zone_probs_rows(cond).numpy()
    assert rows[1].tolist() == [0.0, 0.0, 1.0]            # gamma, gamma primary, high E
    assert rows[0, 0] == 1.0 and rows[2, 0] == 1.0 and rows[3, 0] == 1.0
    assert m.to_generation_config()["zone_primary_energy_edges"] == [0.0]


# ---- energy loss flow of the primary's own crossing (v1.0.0) -------------------------------

def _loss_model():
    return _model(energy_loss_flow=True)


def test_loss_flow_needs_the_atom():
    with pytest.raises(ValueError, match="uncollided_atom"):
        magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
            n_types=N_TYPES, cond_dim=COND_DIM, line_positions_y=LINES, continuum_mode="flow",
            continuum_flow_warp="affine", energy_loss_flow=True)


def test_own_species_continuum_energy_is_below_the_primary_energy():
    m = _loss_model()
    n = 4000
    # own species (gamma crossing of a gamma primary, mu- of a mu-) and the other species
    cond = _cond([0] * n + [1] * n + [1] * n, [0] * n + [1] * n + [0] * n,
                 np.r_[np.full(n, 0.5), np.full(n, 2.0), np.full(n, 2.0)])
    out = m.generate(cond, 3 * n)
    y, comp = out["energy_y"].numpy(), out["energy_component_idx"].numpy()
    ey = m._primary_y(cond).numpy()
    cont = comp == 0
    own = np.r_[np.ones(2 * n, bool), np.zeros(n, bool)]
    assert (y[own & cont] < ey[own & cont]).all()
    assert (own & cont).sum() > 100


def test_loss_flow_trains_only_on_own_species_events():
    m = _loss_model()
    n = 64
    cond = _cond([0] * n + [1] * n, [0] * n + [0] * n, np.full(2 * n, 1.0))   # own, then other species
    ey = m._primary_y(cond).numpy()
    gate = np.zeros((2 * n, N_ZONES), np.float32); gate[:, 0] = 1.0       # all continuum
    y = np.hstack([np.zeros((2 * n, 4), np.float32), (ey - 0.2)[:, None].astype("float32"), gate])
    z = tf.zeros((2 * n, m.latent_dim))

    def grads(sel):
        with tf.GradientTape() as tape:
            _, pieces = m._reconstruction_terms(tf.constant(y[sel]), m.decode(z[: sel.sum()], tf.gather(cond, np.flatnonzero(sel))))
            loss = tf.reduce_mean(pieces["energy_mixture_nll"])
        g = tape.gradient(loss, m.loss_flow.trainable_variables)
        return sum(float(tf.reduce_sum(tf.abs(x))) for x in g if x is not None)

    own = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    assert grads(own) > 0.0 and grads(~own) == 0.0


def test_loss_flow_config_round_trip():
    cfg = _loss_model().to_generation_config()
    assert cfg["energy_loss_flow"] is True and cfg["loss_flow_warp_y_knots"] is None
    json.dumps(cfg)


def test_logit_loss_variable_keeps_energy_below_the_primary_and_round_trips():
    m = _model(energy_loss_flow=True, loss_flow_variable="logit")
    n = 3000
    cond = _cond([0] * n, [0] * n, np.full(n, 1.5))
    out = m.generate(cond, n)
    y, comp = out["energy_y"].numpy(), out["energy_component_idx"].numpy()
    assert (y[comp == 0] < m._primary_y(cond).numpy()[comp == 0]).all()
    assert m.to_generation_config()["loss_flow_variable"] == "logit"
    with pytest.raises(ValueError, match="loss_flow_variable"):
        _model(energy_loss_flow=True, loss_flow_variable="other")


def test_logit_density_integrates_to_one_in_y():
    """The change of variables v(y) must carry its Jacobian: integrate p(y) over y below log10 E_prim."""
    m = _model(energy_loss_flow=True, loss_flow_variable="logit")
    ey = 1.0
    ys = np.linspace(ey - 6.0, ey - 1e-6, 20001).astype("float32")
    n = len(ys)
    cond = _cond([0] * n, [0] * n, np.full(n, ey))
    gate = np.zeros((n, N_ZONES), np.float32); gate[:, 0] = 1.0
    yc = np.hstack([np.zeros((n, 4), np.float32), ys[:, None], gate])
    p = m.decode(tf.zeros((n, m.latent_dim)), cond)
    ey_model = float(m._primary_y(cond).numpy()[0])
    w = np.maximum(ey_model - ys, 1e-7)
    ln_f = -w * np.log(10.0); v = ln_f - np.log(-np.expm1(ln_f))
    lp = m.loss_flow.log_prob(tf.constant(v.astype("float32")), p["flow_cond"]).numpy() + np.log(np.log(10.0)) - np.log(-np.expm1(ln_f))
    integral = np.trapz(np.exp(lp), ys)
    assert 0.95 < integral < 1.02
