## PACKAGE STRUCTURE

GEEANNT/
│
├── __init__.py
├── setup.py
├── README.md
├── GEEANNT.py
│
├── core/
│   ├── __init__.py
│   ├── model.py
│   ├── losses.py
│   ├── geometry.py
│   ├── defaults.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── optuna_phase1.py
│   ├── optuna_phase2.py
│   ├── optuna_phase3.py
│
├── data/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── transforms.py
│
├── utils/
│   ├── __init__.py
│   ├── plotting.py
│   ├── physics.py
│
└── config.py

# GEEANNT

## Geant4 Efficiency Enhancing Artificial Neural Network Toolkit

GEEANNT is a deep-learning framework designed to model and generate particle phase-space distributions from Geant4 Monte Carlo simulations.

The goal of the project is to **replace or augment traditional Monte Carlo particle sources** with fast, learned generative models, enabling significant efficiency improvements in radiation transport simulations.

---

## Motivation

Monte Carlo simulations (e.g. Geant4) are computationally expensive, especially when:

- Large numbers of particles are required
- Rare events must be sampled
- Complex geometries introduce inefficiencies

GEEANNT addresses this by learning the underlying **phase-space distribution** of particles and generating new samples at a fraction of the computational cost.

---

## Core Idea

The framework is based on **Conditional Variational Autoencoders (CVAE)** trained on particle data.

Each event is represented in terms of:

- Energy (`E`)
- Position (`x, y, z`) on a detector surface
- Direction (`vx, vy, vz`)
- Particle type (conditioning variable)

The model learns the joint distribution:

\[
p(E, \vec{r}, \vec{v} \mid \text{particle type})
\]

and can generate new physically consistent samples.

---

## Model Architecture

The current implementation (v1.6) uses:

- Conditional Variational Autoencoder (CVAE)
- Multi-head output structure:
  - **Energy head** (categorical or binned)
  - **Radial position head** (Gaussian)
  - **Angular heads** (cos/sin representation)
  - **Directional variable** (`u_v`, currently discretized or transformed)
- Physics-informed loss terms:
  - Reconstruction losses
  - Angular consistency losses
  - Geometric constraints
  - KL divergence regularization

---

## Training Pipeline

Training is performed in three stages:

### Phase 1 — Random Search (Optuna)

- Wide hyperparameter exploration
- Fast training (few epochs)
- Selection based on surrogate loss

---

### Phase 2 — Physics-aware Optimization

- Narrowed hyperparameter space
- Evaluation based on:
  - Wasserstein distances
  - Physical distributions (x, y, z, directions)
  - Spike penalties

---

### Phase 3 — Final Training

- Full training of top-performing configurations
- Detailed validation against real Monte Carlo data
- Final model selection

---

## Evaluation Metrics

The model is evaluated using:

- Wasserstein distance between real and generated distributions
- Angular reconstruction quality
- Spatial consistency (x, y, z)
- Velocity components (vx, vy, vz)
- Histogram-based spike penalties

---

## Current Limitations

- Trade-off between:
  - Spatial reconstruction (x, y, z)
  - Directional accuracy (vx, vy)
- Instability in transverse components (x, y)
- Discretization of directional variable (`u_v`) limits expressivity
- Model capacity may be insufficient for full phase-space learning

---

## Ongoing Developments

Planned improvements include:

- Conditional Normalizing Flows (CNF)
- Continuous modeling of directional variables
- Improved geometric loss formulations
- Better disentanglement of latent space
- Hybrid CVAE + Flow architectures

---

## Installation

```bash
git clone https://github.com/your-username/GEEANNT.git
cd GEEANNT
pip install .