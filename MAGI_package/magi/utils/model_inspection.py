"""
Model inspection utilities for MAGI.
"""

import tensorflow as tf

def print_model_structure(model):
    print("\n===== MAGI MODEL STRUCTURE =====")
    print(f"Model class   = {model.__class__.__name__}")
    print(f"latent_dim    = {model.latent_dim}")
    print(f"n_types       = {model.n_types}")
    if hasattr(model, "n_energy_bins"):
        print(f"n_energy_bins = {model.n_energy_bins}")
    if hasattr(model, "n_lines"):
        # v0.8 mixture energy head: continuum + fixed lines instead of bins
        print(f"n_lines       = {model.n_lines} (mixture energy head)")
        print(f"continuum     = {getattr(model, 'continuum_mode', 'gaussian')}"
              f" (warp={getattr(model, 'continuum_flow_warp', '-')})")
        print(f"prior         = {getattr(model, 'prior_mode', 'gaussian')}")
    if hasattr(model, "n_uv_bins"):
        print(f"n_uv_bins     = {model.n_uv_bins}")

    if hasattr(model, "y_cont_dim"):
        print(f"y_cont_dim    = {model.y_cont_dim}")

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
            print(f"{k:>8s} : {v}")
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


def _module_param_dict(module):
    trainable = _count_params_from_weights(getattr(module, "trainable_weights", []))
    non_trainable = _count_params_from_weights(getattr(module, "non_trainable_weights", []))
    total = trainable + non_trainable
    status = "trainable" if getattr(module, "trainable", True) else "frozen"
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

    Supports both:
    - CVAE_CatEnergy_CatUV
    - CVAE_CatEnergy_CatUV_TaskAdaptive
    - CVAE_CatEnergy_ContPhi_TaskAdaptive
    """
    print("\n===== MAGI MODEL TREE (WITH PARAMS) =====")
    print(f"{model.__class__.__name__}")
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
    if hasattr(model, "n_lines"):
        # v0.8 mixture energy head
        print(f"│   ├── energy : mixture NLL "
              f"({getattr(model, 'continuum_mode', 'gaussian')} continuum "
              f"+ {model.n_lines} fixed lines)")
        print(f"│   ├── gate   : auxiliary CE "
              f"(w={getattr(model, 'w_gate_aux', 0.0)}, "
              f"focal_gamma={getattr(model, 'gate_focal_gamma', 0.0)})")
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
        print("│   └── phi_v_q: Gaussian NLL")
    else:
        print("│   └── phi_v  : weighted (normalized MSE + angular loss)")

    if hasattr(model, "w_xy"):
        print("│   ├── xy     : geometric xy loss")
    if hasattr(model, "w_vxy"):
        print("│   └── vxy    : physical directional Cartesian loss")

    # ------------------------------------------------------
    # Task weights
    # ------------------------------------------------------
    if hasattr(model, "task_weights"):
        print("│")
        print("├── Active task weights")
        for k, v in model.task_weights.items():
            print(f"│   ├── {k:>8s} : {float(v):.6f}")
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