def print_model_structure(model):
    print("\n===== GEEANNT MODEL STRUCTURE =====")
    print(f"latent_dim = {model.latent_dim}")
    print(f"n_types = {model.n_types}")
    print(f"n_energy_bins = {model.n_energy_bins}")
    print(f"n_uv_bins = {model.n_uv_bins}")

    print("\n--- Encoder ---")
    model.encoder.summary()

    print("\n--- Shared decoder backbone ---")
    model.decoder_backbone.summary()

    print("\n--- Energy branch ---")
    model.energy_branch.summary()

    print("\n--- Position branch ---")
    model.position_branch.summary()

    print("\n--- Direction branch ---")
    model.direction_branch.summary()

    print("\n--- s_r head ---")
    model.sr_head.summary()

    print("\n--- u_v head ---")
    model.uv_head.summary()

    print("\n--- phi_r head ---")
    model.phi_r_head.summary()

    print("\n--- phi_v head ---")
    model.phi_v_head.summary()