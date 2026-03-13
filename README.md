# GEEANNT
Geant4 Efficiency Enhancement Artificial Neural Network Training

A cVAE (Conditional Variational Auto-Encoder) with the purpose of improving Geant4 simulations efficiency, especially in low statistic cases, by sampling and reproducing multi-variate distribution in multi-step simulations.

It's based on ````keras```` and ````tensorflow```` packages in python.

13/03/2026 - State of the art training:
The model is being trained on a dataset of K-40 emissions tracked in the innermost layers of the cryostat for the testing of XFDM (TESs for newATHENA). This dataset is composed by 
```
["EventId", "ParticleName", "Energy", "X", "Y", "Z", "Vx", "Vy", "Vz"]
```
however, when imported it gets separated in the different particles, and the eventIds gets discarded.

The model has the following parameters, depending on the version of the code (each model is well commented):

````
# ==========================================================
# CVAE_CatLogE_GaussSV
# ==========================================================
#
# Conditional Variational Autoencoder with separate decoder heads
# designed to model different variables more appropriately:
#
#   - logE with narrow spectral lines -> discretized categorical head
#   - s_r                             -> Gaussian head
#   - s_v = atanh(u_v)                -> Gaussian head
#   - (cphi_r, sphi_r)                -> 2D head with unit-circle regularization
#   - (cphi_v, sphi_v)                -> 2D head with unit-circle regularization
#
# INPUT / OUTPUT FEATURES
#   y = [logE_s, s_r_s, cphi_r, sphi_r, s_v_s, cphi_v, sphi_v]
#
# where:
#   - logE_s, s_r_s, s_v_s are the scaled versions used during training
#   - cphi_*, sphi_* remain unscaled
#   - u_v = cos(theta_v) is not learned directly
#   - the model instead learns s_v = atanh(u_v), and during generation:
#
#         u_v = tanh(s_v)
#
# ----------------------------------------------------------
# WHY USE s_v INSTEAD OF DIRECT u_v
# ----------------------------------------------------------
#
# u_v is bounded in [-1,1] and in the dataset it contains a large
# concentration of values close to -1.
#
# This makes it difficult to model properly using:
#
#   - a direct Gaussian on u_v (incorrect support)
#   - a simple Beta distribution on x_v=(u_v+1)/2 (which may create
#     artificial spikes at the boundary)
#
# The transformation:
#
#     s_v = atanh(u_v)
#
# maps u_v to an unbounded variable on R, which is better suited
# for a Gaussian head.
#
# In practice this typically:
#
#   - removes artificial spikes at u_v = -1
#   - improves training stability
#   - reduces the spurious collapse of vx and vy toward zero
#
# ----------------------------------------------------------
# NOTE ON MORE EXPRESSIVE HEADS
# ----------------------------------------------------------
#
# In this version:
#
#   - the s_v head is no longer purely linear, but passes through
#     a small dedicated subnetwork. This helps model distributions
#     with slightly multiple peaks, asymmetries, or shapes that
#     cannot be well captured by a single linear mapping from
#     the decoder backbone.
#
#   - the phi_v head is also made more expressive. This helps when
#     cphi_v/sphi_v (and therefore vx, vy) still appear too noisy
#     or unstable.
#
# ----------------------------------------------------------
# MAIN PARAMETERS
# ----------------------------------------------------------
#
# n_types
#   Number of conditioning particle classes.
#
# logE_bin_edges
#   Bin edges used for the categorical logE head.
#
# latent_dim
#   Dimension of the latent bottleneck.
#
# hidden
#   Dense layer structure of the encoder/decoder backbone.
#
# beta
#   Weight of the KL term in the VAE loss.
#
# type_weights
#   Class weights used to compensate particle type imbalance.
#
# min_log_sigma, max_log_sigma
#   Allowed range for the Gaussian head sigmas.
#
# lambda_sigma
#   Penalty applied when sigma becomes too large.
#
# sigma_target
#   Threshold above which log_sigma is penalized.
#
# lambda_phi
#   Regularization strength enforcing unit norm for (cphi_v, sphi_v).
#
# lambda_phi_r
#   Same regularization for (cphi_r, sphi_r).
#
# w_logE
#   Weight of the energy reconstruction loss.
#
# w_sr
#   Weight of the loss on s_r.
#
# w_sv
#   Weight of the loss on s_v.
#
# w_phi_r
#   Weight of the loss on (cphi_r, sphi_r).
#
# w_phi_v
#   Weight of the loss on (cphi_v, sphi_v).
#
# w_xy
#   Weight of the geometric loss directly applied to x,y
#   reconstructed from (s_r, phi_r).
#
# sphere_R
#   Sphere radius used in the geometric xy-loss.
# ==========================================================
````
