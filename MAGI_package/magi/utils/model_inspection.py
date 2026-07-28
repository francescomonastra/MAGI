"""
Model inspection utilities for MAGI.

The printers here are meant to be readable descriptions of what the model
*is*, not just how many weights it has: for the v0.8 mixture-energy head the
architecture is only half the story, since the energy distribution is a
gate-weighted mixture of a normalizing-flow continuum and fixed-position
detector lines, under a learned conditional prior. Those three pieces are
described with their formulas and their actual configured values.
"""

import numpy as np
import tensorflow as tf

# 1 / (2 * sqrt(2 ln 2)) - FWHM to sigma
FWHM_TO_SIGMA = 1.0 / 2.3548200450309493
LN10 = 2.302585092994046


def _is_mixture_energy(model):
    """True for the v0.8 mixture energy head (continuum + fixed lines)."""
    return hasattr(model, "n_lines") and hasattr(model, "line_positions_y")


def _line_table(model):
    """Per-line (label-free) physical summary of the fixed line components.

    Returns a list of (index, E_keV, sigma_y, FWHM_eV). Line positions live in
    y = log10(E / MeV), so a width sigma_y in that space is a *relative* energy
    width: dE/E = ln(10) * sigma_y.
    """
    try:
        pos = np.asarray(model.line_positions_y, dtype=np.float64).reshape(-1)
    except Exception:
        return []
    try:
        logsig = np.asarray(
            model._line_logsigma_clipped().numpy(), dtype=np.float64
        ).reshape(-1)
    except Exception:
        return []

    if logsig.size == 1 and pos.size > 1:
        logsig = np.repeat(logsig, pos.size)

    rows = []
    for i, (y, ls) in enumerate(zip(pos, logsig)):
        e_mev = 10.0 ** y
        sigma_y = float(np.exp(ls))
        fwhm_ev = sigma_y * LN10 * e_mev * 1e6 / FWHM_TO_SIGMA
        rows.append((i, e_mev * 1e3, sigma_y, fwhm_ev))
    return rows


def _print_generative_story(model):
    """The probabilistic model, with the formulas the code actually implements."""
    prior_mode = getattr(model, "prior_mode", "gaussian")
    d = model.latent_dim

    print("\n--- Generative model (what sampling actually does) ---")
    print("  cond c   : particle-type one-hot, dim %d" % model.n_types)
    if prior_mode == "coupling":
        print(f"  1. z ~ p(z | c)      learned coupling-flow prior, z in R^{d}")
    else:
        print(f"  1. z ~ N(0, I)       fixed standard normal prior, z in R^{d}")
    print("  2. h = decoder(z, c) shared stem -> per-variable branches")
    print("  3. y ~ p(y | h)      factorized over the physical variables:")

    if _is_mixture_energy(model):
        print("       energy_y = log10(E/MeV) ~ gate-weighted mixture (below)")
    elif hasattr(model, "n_energy_bins"):
        print(f"       energy   ~ Categorical over {model.n_energy_bins} log-E bins")
    for var, present in (
        ("u_r_q", "ur_mu_head"),
        ("u_v_q", "uv_mu_head"),
        ("phi_r_q", "phi_r_mu_head"),
        ("phi_v_q", "phi_v_mu_head"),
    ):
        if hasattr(model, present):
            print(f"       {var:<8s} ~ N(mu(h), sigma(h))   (quantile-transformed)")
    print("  4. invert the quantile transforms -> physical (x,y,z, vx,vy,vz, E)")


def _print_prior_block(model):
    prior_mode = getattr(model, "prior_mode", "gaussian")
    print("\n--- Latent prior p(z | c) ---")
    if prior_mode != "coupling" or getattr(model, "prior", None) is None:
        print("  mode : gaussian (fixed)")
        print("  p(z|c) = N(0, I), independent of c")
        print("  KL(q||p) is closed-form:")
        print("    KL = -1/2 * sum_d [ 1 + logvar_d - mu_d^2 - exp(logvar_d) ]")
        return

    prior = model.prior
    print("  mode : coupling (learned, conditional)")
    print(f"  {prior.n_layers} affine coupling layers, conditioner hidden="
          f"{tuple(prior.hidden)}, log-scale clamp={prior.log_scale_clamp}")
    print("  Each layer splits z by an alternating binary mask m:")
    print("    z_pass = m * z                       (unchanged, feeds the conditioner)")
    print("    s, t   = MLP([z_pass, c])            (log-scale, shift)")
    print("    s      = clamp * tanh(s_raw)         (keeps exp(s) bounded)")
    print("    z'     = z_pass + (1-m) * (z * exp(s) + t)")
    print("    log|det J| = sum (1-m) * s")
    print("  so   log p(z|c) = log N(f(z;c); 0, I) + sum_layers log|det J|")
    print("  KL is then a 1-sample Monte-Carlo estimate, not closed-form:")
    print("    KL ~= log q(z|x) - log p(z|c),  z the sample already drawn this step")
    print("  Why: a fixed N(0,I) prior forces the *aggregate* posterior to be")
    print("  isotropic, which fights the physical correlations between energy and")
    print("  direction. A learned p(z|c) absorbs them instead.")


def _print_energy_head(model):
    """The v0.8 mixture energy head: gate + continuum + fixed lines."""
    n_lines = int(model.n_lines)
    K = int(getattr(model, "n_continuum_components", 1))
    cont_mode = getattr(model, "continuum_mode", "gaussian")

    print("\n--- Energy head: gated mixture (continuum + fixed lines) ---")
    print(f"  y = log10(E / MeV);  {K} continuum slot(s) + {n_lines} line slot(s)"
          f" = {K + n_lines} gate slots")
    print("    p(y | z, c) = sum_k pi_k * p_cont,k(y)  +  sum_l pi_l * N(y; y_l, sigma_l)")
    print("    pi = softmax(gate_logits(h))          <- learned per event")
    print("  The line positions y_l are FIXED physics inputs (measured line")
    print("  energies mapped into y-space); only their mixture weights are learned.")

    # ---- gate
    print("\n  Gate:")
    w_aux = float(getattr(model, "w_gate_aux", 0.0))
    gamma = float(getattr(model, "gate_focal_gamma", 0.0))
    print(f"    auxiliary CE weight w_gate_aux = {w_aux}")
    print(f"    focal exponent gamma           = {gamma}")
    if gamma > 0:
        print("      L_gate = -sum_j t_j * (1 - pi_j)^gamma * log pi_j")
        print("      the (1-pi)^gamma factor down-weights the easy ~99% continuum")
        print("      majority so the rare line slots are not drowned out.")
    else:
        print("      L_gate = -sum_j t_j * log pi_j    (plain CE, no focal term)")
    gcw = getattr(model, "gate_class_weights", None)
    if gcw is not None:
        try:
            gcw = np.asarray(gcw).reshape(-1)
            print(f"      per-slot class weights: {np.round(gcw, 3).tolist()}")
        except Exception:
            pass
    else:
        print("      per-slot class weights: none (all slots weighted equally)")
    if K > 1:
        print(f"      NOTE: the gate has {K}+{n_lines} = {K + n_lines} slots but the")
        print(f"      targets/class weights have {n_lines}+1 columns - the K continuum")
        print("      slots share one target column, so the split between them is")
        print("      unsupervised (that is what continuum_balance is for).")
    print("    Targets t come from build_gate_targets - physical proximity to a")
    print("    known line, computed from the data, never from the model. Without")
    print("    that supervision the encoder leaks 'which line' into z and the gate")
    print("    never learns to use the line slots at all.")

    # ---- continuum
    print("\n  Continuum:")
    if cont_mode == "flow":
        flow = getattr(model, "continuum_flow", None)
        cond = getattr(model, "energy_flow_condition", "z_cond")
        print("    mode: normalizing flow (conditional rational-quadratic spline)")
        if flow is not None:
            print(f"      {flow.n_transforms} stacked spline transform(s) x "
                  f"{flow.n_bins} bins, domain [-{flow.B}, {flow.B}]")
            print(f"      conditioned on: {cond}")
            print(f"      standardization: warp = {flow.warp_mode}")
            if flow.warp_mode == "cdf" and flow.warp_y_knots_np is not None:
                yk = flow.warp_y_knots_np
                print(f"        {yk.size} monotone knots over y in "
                      f"[{yk[0]:.2f}, {yk[-1]:.2f}]")
                print("        w = CDF warp(y): maps the empirical y-marginal to ~N(0,1),")
                print("        so spline knots are density-proportional - resolution")
                print("        goes where the events are, not where the axis is linear.")
            else:
                print("        w = (y - y_mean) / y_scale   (constant Jacobian)")
        print("      log p_cont(y) = log N(g(w); 0,1) + log|dg/dw| + log|dw/dy|")
        print("      Monotone splines are invertible by construction, so the same")
        print("      object gives exact density for training and exact sampling.")
    else:
        print(f"    mode: {K} Gaussian component(s), N(mu_k(h), sigma_k(h))")
        if K > 1:
            wb = float(getattr(model, "w_continuum_balance", 0.0))
            print(f"      balance regularizer w={wb}: penalizes the batch-mean usage")
            print("      of the K slots for deviating from uniform, so one slot cannot")
            print("      absorb all the mass while the others collapse.")

    # ---- lines
    rows = _line_table(model)
    if rows:
        print("\n  Fixed line components:")
        trainable = getattr(model, "line_logsigma_trainable", None)
        width_note = ("learned" if trainable else "pinned")
        print(f"    idx     E [keV]     sigma_y      FWHM [eV]   ({width_note} widths)")
        for i, e_kev, sigma_y, fwhm_ev in rows:
            print(f"    {i:>3d}  {e_kev:>10.4f}  {sigma_y:>10.3e}  {fwhm_ev:>10.2f}")
        print("    sigma_y is a width in log10(E); the physical relative width is")
        print("    dE/E = ln(10) * sigma_y, hence FWHM = 2.355 * ln(10) * sigma_y * E.")

    # ---- regularizers (only those actually active in this configuration)
    rows = _regularizer_rows(model)
    print("\n  Active regularizers:")
    if rows:
        for label, w, note in rows:
            print(f"    {label} (w={w}): {note}")
        print("    Purpose: stop the continuum from quietly reproducing a peak that")
        print("    the dedicated line slots are supposed to own.")
    else:
        print("    none - every continuum/line separation regularizer is a no-op in")
        print("    this configuration (continuum_repulsion has no per-sample mean to")
        print("    repel in flow mode; continuum_balance needs K > 1). Line/continuum")
        print("    separation rests entirely on the gate supervision.")


def _print_objective(model):
    print("\n--- Training objective ---")
    print("  L = rec + beta * KL + regularizers")
    print(f"    beta = {float(getattr(model, 'beta', 0.0))}")
    if _is_mixture_energy(model):
        tw = getattr(model, "task_weights", {})
        print("    rec = w_energy * mixture_NLL")
        print("        + w_gate_aux * gate_CE")
        print("        + w_ur * ur_NLL + w_uv * uv_NLL")
        print("        + w_phi_r * phi_r_NLL + w_phi_v * phi_v_NLL")
        print("    regularizers = sigma_reg")
        for label, w, _note in _regularizer_rows(model):
            print(f"                 + {w} * {label}")
        print("\n    current weights:")
        for k, v in tw.items():
            print(f"      w_{k:<8s} = {float(v):.6f}")
        print(f"      w_gate_aux = {float(getattr(model, 'w_gate_aux', 0.0)):.6f}")
        print("\n    Task weights are tf.Variables, so a mid-fit change (e.g. from")
        print("    TaskAdaptiveLossScheduler.decay_task_weight) reaches the already-")
        print("    compiled train_step graph immediately - no retrace needed.")
    else:
        print("    rec = sum of the per-variable weighted losses listed below")


def print_model_structure(model, explain=True, summaries=True):
    """Describe the model: configuration, probabilistic structure, Keras summaries.

    Parameters
    ----------
    explain : bool
        Include the generative-model / prior / energy-head / objective blocks
        with their formulas. Set False for the bare configuration + summaries.
    summaries : bool
        Include the per-submodule `keras.Model.summary()` dumps.
    """
    print("\n===== MAGI MODEL STRUCTURE =====")
    print(f"Model class   = {model.__class__.__name__}")
    print(f"latent_dim    = {model.latent_dim}")
    print(f"n_types       = {model.n_types}")
    if hasattr(model, "n_energy_bins"):
        print(f"n_energy_bins = {model.n_energy_bins}")
    if _is_mixture_energy(model):
        # v0.8 mixture energy head: continuum + fixed lines instead of bins
        print(f"n_lines       = {model.n_lines} (mixture energy head)")
        print(f"continuum     = {getattr(model, 'continuum_mode', 'gaussian')}"
              f" (warp={getattr(model, 'continuum_flow_warp', '-')},"
              f" K={getattr(model, 'n_continuum_components', 1)})")
        print(f"prior         = {getattr(model, 'prior_mode', 'gaussian')}")
    if hasattr(model, "n_uv_bins"):
        print(f"n_uv_bins     = {model.n_uv_bins}")

    if hasattr(model, "y_cont_dim"):
        print(f"y_cont_dim    = {model.y_cont_dim}")

    if explain:
        _print_generative_story(model)
        _print_prior_block(model)
        if _is_mixture_energy(model):
            _print_energy_head(model)
        _print_objective(model)

    if not summaries:
        return

    print("\n--- Encoder ---")
    model.encoder.summary()

    # ------------------------------------------------------
    # Decoder main blocks
    # ------------------------------------------------------
    if hasattr(model, "decoder_backbone"):
        print("\n--- Shared decoder backbone ---")
        model.decoder_backbone.summary()

    if hasattr(model, "decoder_stem"):
        print("\n--- Decoder stem ---")
        model.decoder_stem.summary()

    if hasattr(model, "decoder_deep_trunk"):
        print("\n--- Decoder deep trunk ---")
        model.decoder_deep_trunk.summary()

    # ------------------------------------------------------
    # Branches
    # ------------------------------------------------------
    if hasattr(model, "energy_branch"):
        print("\n--- Energy branch ---")
        model.energy_branch.summary()

    if hasattr(model, "energy_cont_head"):
        print("\n--- Energy continuum feature head ---")
        try:
            model.energy_cont_head.summary()
        except Exception:
            print(model.energy_cont_head)

    if getattr(model, "continuum_flow", None) is not None:
        flow = model.continuum_flow
        print("\n--- Continuum flow (ConditionalRQSFlow) ---")
        print(f"  {flow.n_transforms} transform(s) x {flow.n_bins} bins "
              f"-> {flow._n_params} spline params per event, "
              f"emitted by a conditioner MLP over the {flow.feat_dim}-d feature")
        print(f"  params: {_count_params_from_weights(flow.weights)}")

    if getattr(model, "prior", None) is not None:
        prior = model.prior
        print("\n--- Latent prior (ConditionalCouplingPrior) ---")
        print(f"  {prior.n_layers} coupling layers over z in R^{prior.latent_dim}, "
              f"cond_dim={prior.cond_dim}")
        print(f"  params: {_count_params_from_weights(prior.weights)}")

    if hasattr(model, "position_branch"):
        print("\n--- Position branch ---")
        model.position_branch.summary()

    if hasattr(model, "direction_branch"):
        print("\n--- Direction branch ---")
        model.direction_branch.summary()

    # ------------------------------------------------------
    # Heads
    # ------------------------------------------------------
    if hasattr(model, "sr_head"):
        print("\n--- s_r head ---")
        model.sr_head.summary()

    if hasattr(model, "uv_head"):
        print("\n--- u_v head ---")
        model.uv_head.summary()

    if hasattr(model, "phi_r_head"):
        print("\n--- phi_r head ---")
        model.phi_r_head.summary()

    if hasattr(model, "phi_v_head"):
        print("\n--- phi_v head ---")
        model.phi_v_head.summary()

    if hasattr(model, "ur_head"):
        print("\n--- u_r_q head ---")
        model.ur_head.summary()

    if hasattr(model, "ur_mu_head"):
        print("\n--- ur_mu_head ---")
        print(model.ur_mu_head)

    if hasattr(model, "ur_logsigma_head"):
        print("\n--- ur_logsigma_head ---")
        print(model.ur_logsigma_head)

    if hasattr(model, "uv_mu_head"):
        print("\n--- uv_mu_head ---")
        print(model.uv_mu_head)

    if hasattr(model, "uv_logsigma_head"):
        print("\n--- uv_logsigma_head ---")
        print(model.uv_logsigma_head)

    if hasattr(model, "phi_r_mu_head"):
        print("\n--- phi_r_mu_head ---")
        print(model.phi_r_mu_head)

    if hasattr(model, "phi_r_logsigma_head"):
        print("\n--- phi_r_logsigma_head ---")
        print(model.phi_r_logsigma_head)

    if hasattr(model, "phi_v_mu_head"):
        print("\n--- phi_v_mu_head ---")
        print(model.phi_v_mu_head)

    if hasattr(model, "phi_v_logsigma_head"):
        print("\n--- phi_v_logsigma_head ---")
        print(model.phi_v_logsigma_head)

    # ------------------------------------------------------
    # Final scalar heads
    # ------------------------------------------------------
    if hasattr(model, "energy_logits_head"):
        print("\n--- energy_logits_head ---")
        print(model.energy_logits_head)

    if hasattr(model, "energy_gate_head"):
        print("\n--- energy_gate_head (mixture gate logits) ---")
        print(model.energy_gate_head)

    if hasattr(model, "sr_mu_head"):
        print("\n--- sr_mu_head ---")
        print(model.sr_mu_head)

    if hasattr(model, "sr_logsigma_head"):
        print("\n--- sr_logsigma_head ---")
        print(model.sr_logsigma_head)

    if hasattr(model, "uv_logits_head"):
        print("\n--- uv_logits_head ---")
        print(model.uv_logits_head)

    # ------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------
    print("\n--- Task / loss weights ---")
    if hasattr(model, "task_weights"):
        for k, v in model.task_weights.items():
            print(f"{k:>8s} : {float(v)}")
    else:
        for name in ["w_energy", "w_sr", "w_uv", "w_phi_r", "w_phi_v", "w_xy", "w_vxy", "w_ur"]:
            if hasattr(model, name):
                print(f"{name:>8s} : {getattr(model, name)}")



def _module_status_line(name, module):

    trainable = getattr(module, "trainable", None)

    try:

        n_trainable = _count_params_from_weights(module.trainable_weights)

    except Exception:

        n_trainable = 0

    try:

        n_non_trainable = _count_params_from_weights(module.non_trainable_weights)

    except Exception:

        n_non_trainable = 0

    try:

        n_total = module.count_params()

    except Exception:

        n_total = n_trainable + n_non_trainable

    status = "trainable" if trainable else "frozen"

    return (

        f"{name:<22s} | {status:<9s} | "

        f"params total={n_total:<8d} | "

        f"trainable={n_trainable:<8d} | "

        f"non_trainable={n_non_trainable:<8d}"

    )

def print_trainable_status(model):

    """

    Print trainable/frozen status and parameter counts of the main model blocks.

    Works for both the standard and task-adaptive CVAE variants.

    """

    print("\n===== MAGI TRAINABLE STATUS =====")

    print(f"Model class: {model.__class__.__name__}")

    modules = []

    # Core

    if hasattr(model, "encoder"):

        modules.append(("encoder", model.encoder))

    if hasattr(model, "decoder_backbone"):

        modules.append(("decoder_backbone", model.decoder_backbone))

    if hasattr(model, "decoder_stem"):

        modules.append(("decoder_stem", model.decoder_stem))

    if hasattr(model, "decoder_deep_trunk"):

        modules.append(("decoder_deep_trunk", model.decoder_deep_trunk))

    # Branches

    if hasattr(model, "energy_branch"):

        modules.append(("energy_branch", model.energy_branch))

    if hasattr(model, "position_branch"):

        modules.append(("position_branch", model.position_branch))

    if hasattr(model, "direction_branch"):

        modules.append(("direction_branch", model.direction_branch))

    # Heads

    if hasattr(model, "sr_head"):

        modules.append(("sr_head", model.sr_head))

    if hasattr(model, "sr_mu_head"):

        modules.append(("sr_mu_head", model.sr_mu_head))

    if hasattr(model, "sr_logsigma_head"):

        modules.append(("sr_logsigma_head", model.sr_logsigma_head))

    if hasattr(model, "uv_head"):

        modules.append(("uv_head", model.uv_head))

    if hasattr(model, "uv_logits_head"):

        modules.append(("uv_logits_head", model.uv_logits_head))

    if hasattr(model, "phi_r_head"):

        modules.append(("phi_r_head", model.phi_r_head))

    if hasattr(model, "phi_v_head"):

        modules.append(("phi_v_head", model.phi_v_head))

    if hasattr(model, "energy_logits_head"):

        modules.append(("energy_logits_head", model.energy_logits_head))

    # v0.8 mixture energy head + learned prior
    for attr in ("energy_gate_head", "energy_cont_head", "energy_cont_mu_head",
                 "energy_cont_logsigma_head", "continuum_flow", "prior"):
        mod = getattr(model, attr, None)
        if mod is not None:
            modules.append((attr, mod))

    if hasattr(model, "ur_head"):
        modules.append(("ur_head", model.ur_head))

    if hasattr(model, "ur_mu_head"):
        modules.append(("ur_mu_head", model.ur_mu_head))

    if hasattr(model, "ur_logsigma_head"):
        modules.append(("ur_logsigma_head", model.ur_logsigma_head))

    if hasattr(model, "uv_mu_head"):
        modules.append(("uv_mu_head", model.uv_mu_head))

    if hasattr(model, "uv_logsigma_head"):
        modules.append(("uv_logsigma_head", model.uv_logsigma_head))

    if hasattr(model, "phi_r_mu_head"):
        modules.append(("phi_r_mu_head", model.phi_r_mu_head))

    if hasattr(model, "phi_r_logsigma_head"):
        modules.append(("phi_r_logsigma_head", model.phi_r_logsigma_head))

    if hasattr(model, "phi_v_mu_head"):
        modules.append(("phi_v_mu_head", model.phi_v_mu_head))

    if hasattr(model, "phi_v_logsigma_head"):
        modules.append(("phi_v_logsigma_head", model.phi_v_logsigma_head))

    for name, module in modules:

        print(_module_status_line(name, module))

    print("\n--- Whole model ---")

    total_trainable = _count_params_from_weights(model.trainable_weights)

    total_non_trainable = _count_params_from_weights(model.non_trainable_weights)

    try:

        total = model.count_params()

    except Exception:

        total = total_trainable + total_non_trainable

    print(

        f"{'model':<22s} | {'mixed':<9s} | "

        f"params total={total:<8d} | "

        f"trainable={total_trainable:<8d} | "

        f"non_trainable={total_non_trainable:<8d}"

    )

    if hasattr(model, "task_weights"):

        print("\n--- Current task weights ---")

        for k, v in model.task_weights.items():

            print(f"{k:>8s} : {float(v):.6f}")



def _count_params_from_weights(weights):
    total = 0
    for w in weights:
        n = 1
        for d in w.shape:
            n *= int(d)
        total += n
    return total


def _regularizer_rows(model):
    """(label, weight, note) for the regularizers that are actually active.

    Several are deliberate no-ops in some configurations - continuum_repulsion
    has no per-sample mean to repel in flow mode, and continuum_balance has
    nothing to balance with a single continuum slot - so listing them with
    their nonzero weight would misreport what the objective contains.
    """
    K = int(getattr(model, "n_continuum_components", 1))
    flow = getattr(model, "continuum_mode", "gaussian") == "flow"
    rows = []

    w = float(getattr(model, "w_continuum_repulsion", 0.0))
    if w and not flow:
        margin = float(getattr(model, "continuum_repulsion_margin", 0.0))
        rows.append(("continuum_repulsion", w,
                     f"keep the continuum mean >{margin} from any line position"))

    w = float(getattr(model, "w_flow_line_repulsion", 0.0))
    if w and flow:
        rows.append(("flow_line_repulsion", w,
                     "keep flow density off the pinned line positions"))

    w = float(getattr(model, "w_continuum_balance", 0.0))
    if w and K > 1:
        rows.append(("continuum_balance", w,
                     f"keep the {K} continuum slots near uniform usage"))

    return rows


def _module_param_dict(module):
    trainable = _count_params_from_weights(getattr(module, "trainable_weights", []))
    non_trainable = _count_params_from_weights(getattr(module, "non_trainable_weights", []))
    total = trainable + non_trainable
    if not getattr(module, "trainable", True):
        status = "frozen"
    elif total == 0 and not getattr(module, "built", True):
        # A Dense layer built lazily on first call has no weights yet, which
        # would otherwise be reported as a genuine 0-parameter block. Layers
        # that own sublayers created in __init__ report built=False while
        # already holding weights, so require total == 0 as well.
        status = "unbuilt"
    else:
        status = "trainable"
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "status": status,
    }


def _fmt_module_line(name, module, prefix="", connector="├── "):
    info = _module_param_dict(module)
    return (
        f"{prefix}{connector}{name} "
        f"[{info['status']}] "
        f"(total={info['total']}, "
        f"trainable={info['trainable']}, "
        f"non_trainable={info['non_trainable']})"
    )


def print_model_tree_with_params(model):
    """
    Print a graphical ASCII tree of the model architecture, including:
    - branching structure
    - trainable/frozen status
    - parameter counts per block

    Supports:
    - CVAE_CatEnergy_CatUV
    - CVAE_CatEnergy_CatUV_TaskAdaptive
    - CVAE_CatEnergy_ContPhi_TaskAdaptive
    - CVAE_MixEnergy_ContPhi_TaskAdaptive (v0.8: gate + flow continuum +
      fixed lines + learned conditional prior, all shown with their formulas)
    """
    print("\n===== MAGI MODEL TREE (WITH PARAMS) =====")
    print(f"{model.__class__.__name__}")
    if _is_mixture_energy(model):
        print("  p(y|z,c): gated mixture of a continuum density and "
              f"{int(model.n_lines)} fixed detector lines")
        print(f"  p(z|c)  : {getattr(model, 'prior_mode', 'gaussian')}")
    print("│")

    # ------------------------------------------------------
    # Encoder
    # ------------------------------------------------------
    if hasattr(model, "encoder"):
        print(_fmt_module_line("Encoder", model.encoder, prefix="", connector="├── "))
        if hasattr(model, "n_uv_bins"):
            print("│   ├── input: [y_cont, E_onehot, uv_onehot, cond]")
        else:
            print("│   ├── input: [y_cont, E_onehot, cond]")
        print("│   ├── z_mean")
        print("│   └── z_logvar")
        print("│")

    # ------------------------------------------------------
    # Standard decoder structure
    # ------------------------------------------------------
    if hasattr(model, "decoder_backbone"):
        print(_fmt_module_line("Decoder backbone", model.decoder_backbone, prefix="", connector="├── "))
        print("│   │")

        if hasattr(model, "energy_branch"):
            print(_fmt_module_line("Energy branch", model.energy_branch, prefix="│   ", connector="├── "))
            if hasattr(model, "energy_logits_head"):
                print(_fmt_module_line("energy_logits_head", model.energy_logits_head, prefix="│   │   ", connector="└── "))

        if hasattr(model, "position_branch"):
            print(_fmt_module_line("Position branch", model.position_branch, prefix="│   ", connector="├── "))
            if hasattr(model, "sr_head"):
                print(_fmt_module_line("sr_head", model.sr_head, prefix="│   │   ", connector="├── "))
                if hasattr(model, "sr_mu_head"):
                    print(_fmt_module_line("sr_mu_head", model.sr_mu_head, prefix="│   │   │   ", connector="├── "))
                if hasattr(model, "sr_logsigma_head"):
                    print(_fmt_module_line("sr_logsigma_head", model.sr_logsigma_head, prefix="│   │   │   ", connector="└── "))
                if hasattr(model, "phi_r_head"):
                    connector = "└── " if not hasattr(model, "phi_r_mu_head") else "├── "
                    print(_fmt_module_line("phi_r_head", model.phi_r_head, prefix="│   │   ", connector=connector))

                    if hasattr(model, "phi_r_mu_head"):
                        print(_fmt_module_line("phi_r_mu_head", model.phi_r_mu_head, prefix="│   │   │   ", connector="├── "))

                    if hasattr(model, "phi_r_logsigma_head"):
                        print(_fmt_module_line("phi_r_logsigma_head", model.phi_r_logsigma_head, prefix="│   │   │   ", connector="└── "))

        if hasattr(model, "direction_branch"):
            print(_fmt_module_line("Direction branch", model.direction_branch, prefix="│   ", connector="└── "))
            if hasattr(model, "uv_head"):
                print(_fmt_module_line("uv_head", model.uv_head, prefix="│       ", connector="├── "))
                if hasattr(model, "uv_logits_head"):
                    print(_fmt_module_line("uv_logits_head", model.uv_logits_head, prefix="│       │   ", connector="└── "))
                if hasattr(model, "phi_v_head"):
                    connector = "└── " if not hasattr(model, "phi_v_mu_head") else "├── "
                    print(_fmt_module_line("phi_v_head", model.phi_v_head, prefix="│       ", connector=connector))

                    if hasattr(model, "phi_v_mu_head"):
                        print(_fmt_module_line("phi_v_mu_head", model.phi_v_mu_head, prefix="│       │   ", connector="├── "))

                    if hasattr(model, "phi_v_logsigma_head"):
                        print(_fmt_module_line("phi_v_logsigma_head", model.phi_v_logsigma_head, prefix="│       │   ", connector="└── "))

    # ------------------------------------------------------
    # Task-adaptive decoder structure
    # ------------------------------------------------------
    elif hasattr(model, "decoder_stem") and hasattr(model, "decoder_deep_trunk"):
        print(_fmt_module_line("Decoder stem", model.decoder_stem, prefix="", connector="├── "))
        print("│   └── light shared representation")
        print("│")

        if hasattr(model, "energy_branch"):
            print(_fmt_module_line("Energy branch (short path)", model.energy_branch, prefix="", connector="├── "))
            if hasattr(model, "energy_logits_head"):
                print(_fmt_module_line("energy_logits_head", model.energy_logits_head, prefix="│   ", connector="└── "))

            # v0.8 mixture head: gate over (K continuum + n_lines) slots, a
            # continuum density (flow or Gaussian), and fixed line components.
            if _is_mixture_energy(model):
                K = int(getattr(model, "n_continuum_components", 1))
                n_lines = int(model.n_lines)
                if hasattr(model, "energy_gate_head"):
                    print(_fmt_module_line("energy_gate_head", model.energy_gate_head,
                                           prefix="│   ", connector="├── "))
                    print(f"│   │   └── pi = softmax(.) over {K}+{n_lines}"
                          f" = {K + n_lines} slots")
                if hasattr(model, "energy_cont_head"):
                    print(_fmt_module_line("energy_cont_head", model.energy_cont_head,
                                           prefix="│   ", connector="├── "))
                if getattr(model, "continuum_flow", None) is not None:
                    flow = model.continuum_flow
                    print(_fmt_module_line("continuum_flow (RQS)", flow,
                                           prefix="│   ", connector="├── "))
                    print(f"│   │   ├── {flow.n_transforms} x {flow.n_bins}-bin spline,"
                          f" domain [-{flow.B}, {flow.B}], warp={flow.warp_mode}")
                    print(f"│   │   └── cond: {getattr(model, 'energy_flow_condition', '-')}"
                          f"  ->  log p_cont(y) = log N(g(w);0,1) + log|dg/dw| + log|dw/dy|")
                else:
                    if hasattr(model, "energy_cont_mu_head"):
                        print(_fmt_module_line("energy_cont_mu_head", model.energy_cont_mu_head,
                                               prefix="│   ", connector="├── "))
                    if hasattr(model, "energy_cont_logsigma_head"):
                        print(_fmt_module_line("energy_cont_logsigma_head",
                                               model.energy_cont_logsigma_head,
                                               prefix="│   ", connector="├── "))
                rows = _line_table(model)
                if rows:
                    e_lo = min(r[1] for r in rows)
                    e_hi = max(r[1] for r in rows)
                    trainable = getattr(model, "line_logsigma_trainable", None)
                    print(f"│   └── {n_lines} fixed line component(s) "
                          f"[{'learned' if trainable else 'pinned'} width] "
                          f"({e_lo:.3f} - {e_hi:.3f} keV, 0 trainable positions)")
        print("│")

        if getattr(model, "prior", None) is not None:
            print(_fmt_module_line("Latent prior p(z|c)", model.prior,
                                   prefix="", connector="├── "))
            print(f"│   └── {model.prior.n_layers} affine coupling layers; "
                  f"KL is 1-sample MC: log q(z|x) - log p(z|c)")
            print("│")
        elif _is_mixture_energy(model):
            print("├── Latent prior p(z|c) = N(0, I) [fixed, 0 params]")
            print("│   └── KL closed-form: -1/2 sum_d [1 + logvar - mu^2 - exp(logvar)]")
            print("│")

        print(_fmt_module_line("Decoder deep trunk", model.decoder_deep_trunk, prefix="", connector="├── "))
        print("│   └── deep shared representation")

        if hasattr(model, "position_branch"):
            print(_fmt_module_line("Position branch", model.position_branch, prefix="│   ", connector="├── "))
            if hasattr(model, "ur_head"):
                print(_fmt_module_line("ur_head", model.ur_head, prefix="│   │   ", connector="├── "))
                if hasattr(model, "ur_mu_head"):
                    print(_fmt_module_line("ur_mu_head", model.ur_mu_head, prefix="│   │   │   ", connector="├── "))
                if hasattr(model, "ur_logsigma_head"):
                    print(_fmt_module_line("ur_logsigma_head", model.ur_logsigma_head, prefix="│   │   │   ", connector="└── "))
            if hasattr(model, "sr_head"):
                print(_fmt_module_line("sr_head", model.sr_head, prefix="│   │   ", connector="├── "))
                if hasattr(model, "sr_mu_head"):
                    print(_fmt_module_line("sr_mu_head", model.sr_mu_head, prefix="│   │   │   ", connector="├── "))
                if hasattr(model, "sr_logsigma_head"):
                    print(_fmt_module_line("sr_logsigma_head", model.sr_logsigma_head, prefix="│   │   │   ", connector="└── "))
            if hasattr(model, "phi_r_head"):
                connector = "└── " if not hasattr(model, "phi_r_mu_head") else "├── "
                print(_fmt_module_line("phi_r_head", model.phi_r_head, prefix="│   │   ", connector=connector))

                if hasattr(model, "phi_r_mu_head"):
                    print(_fmt_module_line("phi_r_mu_head", model.phi_r_mu_head, prefix="│   │   │   ", connector="├── "))

                if hasattr(model, "phi_r_logsigma_head"):
                    print(_fmt_module_line("phi_r_logsigma_head", model.phi_r_logsigma_head, prefix="│   │   │   ", connector="└── "))

        if hasattr(model, "direction_branch"):
            print(_fmt_module_line("Direction branch", model.direction_branch, prefix="│   ", connector="└── "))
            if hasattr(model, "uv_head"):
                print(_fmt_module_line("uv_head", model.uv_head, prefix="│       ", connector="├── "))

                if hasattr(model, "uv_logits_head"):
                    print(_fmt_module_line("uv_logits_head", model.uv_logits_head, prefix="│       │   ", connector="└── "))

                if hasattr(model, "uv_mu_head"):
                    print(_fmt_module_line("uv_mu_head", model.uv_mu_head, prefix="│       │   ", connector="├── "))

                if hasattr(model, "uv_logsigma_head"):
                    print(_fmt_module_line("uv_logsigma_head", model.uv_logsigma_head, prefix="│       │   ", connector="└── "))
                    
            if hasattr(model, "phi_v_head"):
                connector = "└── " if not hasattr(model, "phi_v_mu_head") else "├── "
                print(_fmt_module_line("phi_v_head", model.phi_v_head, prefix="│       ", connector=connector))

                if hasattr(model, "phi_v_mu_head"):
                    print(_fmt_module_line("phi_v_mu_head", model.phi_v_mu_head, prefix="│       │   ", connector="├── "))

                if hasattr(model, "phi_v_logsigma_head"):
                    print(_fmt_module_line("phi_v_logsigma_head", model.phi_v_logsigma_head, prefix="│       │   ", connector="└── "))

    else:
        print("├── [Unknown decoder structure]")

    # ------------------------------------------------------
    # Losses
    # ------------------------------------------------------
    print("│")
    print("├── Losses")
    if _is_mixture_energy(model):
        # v0.8 mixture energy head
        print(f"│   ├── energy : mixture NLL "
              f"({getattr(model, 'continuum_mode', 'gaussian')} continuum "
              f"+ {model.n_lines} fixed lines)")
        print("│   │   └── -log[ sum_k pi_k p_cont,k(y) + sum_l pi_l N(y; y_l, sigma_l) ]")
        gamma = float(getattr(model, "gate_focal_gamma", 0.0))
        print(f"│   ├── gate   : auxiliary CE "
              f"(w={getattr(model, 'w_gate_aux', 0.0)}, focal_gamma={gamma})")
        if gamma > 0:
            print("│   │   └── -sum_j t_j (1 - pi_j)^gamma log pi_j   (t from "
                  "build_gate_targets)")
        else:
            print("│   │   └── -sum_j t_j log pi_j   (t from build_gate_targets)")
        short = {"continuum_repulsion": "cont_rep",
                 "flow_line_repulsion": "flow_rep",
                 "continuum_balance": "cont_bal"}
        for label, w, note in _regularizer_rows(model):
            print(f"│   ├── {short.get(label, label):<8s}: w={w} - {note}")
    else:
        print("│   ├── energy : categorical CE")

    if hasattr(model, "ur_mu_head"):
        print("│   ├── ur_q   : Gaussian NLL")
    else:
        print("│   ├── sr     : Gaussian NLL")

    if hasattr(model, "uv_logits_head"):
        print("│   ├── uv     : smoothed categorical CE")
    elif hasattr(model, "uv_mu_head"):
        print("│   ├── uv_q   : Gaussian NLL")

    if hasattr(model, "phi_r_mu_head"):
        print("│   ├── phi_r_q: Gaussian NLL")
    else:
        print("│   ├── phi_r  : MSE")

    if hasattr(model, "phi_v_mu_head"):
        print("│   ├── phi_v_q: Gaussian NLL")
    else:
        print("│   ├── phi_v  : weighted (normalized MSE + angular loss)")

    if hasattr(model, "w_xy"):
        print("│   ├── xy     : geometric xy loss")
    if hasattr(model, "w_vxy"):
        print("│   └── vxy    : physical directional Cartesian loss")

    beta = float(getattr(model, "beta", 0.0))
    if getattr(model, "prior", None) is not None:
        print(f"│   └── KL     : beta={beta} x MC estimate log q(z|x) - log p(z|c)")
    else:
        print(f"│   └── KL     : beta={beta} x closed-form KL(q(z|x) || N(0,I))")

    # ------------------------------------------------------
    # Task weights
    # ------------------------------------------------------
    if hasattr(model, "task_weights"):
        print("│")
        print("├── Active task weights")
        for k, v in model.task_weights.items():
            print(f"│   ├── {k:>8s} : {float(v):.6f}")
        if _is_mixture_energy(model):
            # not in task_weights, but it multiplies a reconstruction term
            print(f"│   ├── {'gate_aux':>8s} : "
                  f"{float(getattr(model, 'w_gate_aux', 0.0)):.6f}")
    else:
        print("│")
        print("├── Static task weights")
        for name in ["w_energy", "w_sr", "w_uv", "w_phi_r", "w_phi_v", "w_xy", "w_ur", "w_vxy"]:
            if hasattr(model, name):
                print(f"│   ├── {name:>8s} : {float(getattr(model, name)):.6f}")

    # ------------------------------------------------------
    # Whole model totals
    # ------------------------------------------------------
    total_trainable = _count_params_from_weights(model.trainable_weights)
    total_non_trainable = _count_params_from_weights(model.non_trainable_weights)
    total = total_trainable + total_non_trainable

    print("│")
    print("└── Model totals")
    print(f"    ├── total params        : {total}")
    print(f"    ├── trainable params    : {total_trainable}")
    print(f"    └── non-trainable params: {total_non_trainable}")


def print_duplicate_trainable_variables(model):
    """
    Print duplicate trainable variables, if any.
    Useful for debugging nested subclassed models.
    """
    print("\n===== DUPLICATE TRAINABLE VARIABLES CHECK =====")

    seen = {}
    duplicates = []

    for i, v in enumerate(model.trainable_variables):
        vid = id(v)
        name = getattr(v, "name", f"var_{i}")

        if vid in seen:
            duplicates.append((i, seen[vid], name))
        else:
            seen[vid] = i

    if not duplicates:
        print("No duplicate trainable variables found.")
        return

    print(f"Found {len(duplicates)} duplicate entries:")
    for i, j, name in duplicates:
        print(f"  duplicate variable at indices {j} and {i}: {name}")