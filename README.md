# Oscillating Cylinder Flow: Neural ODEs vs PINNs

Comparison of machine learning approaches for reconstructing and predicting velocity fields in oscillating cylinder flow.

## Project Overview

This repository contains code for the ME5510 (Scientific Machine Learning) final project. Three methods are compared:

1. **Physics-Informed Neural Networks (PINNs)** - Embed Navier-Stokes equations in loss function
2. **Neural ODE + POD** - Learn dynamics in POD-reduced latent space
3. **Neural ODE + VAE** - Learn dynamics in VAE-encoded latent space

## Results Summary

| Method | f₀=0.1 Hz | f₀=0.3 Hz | f₀=0.2 Hz (unseen) | Training Time |
|--------|-----------|-----------|---------------------|---------------|
| PINN v2 | 3.86% | 10.62% | 18.80% | ~4.5 hrs |
| NODE + POD | 5.65% | 4.77% | 101.73% | ~15 min |
| NODE + VAE | 2.08% | 1.32% | 25.08% | ~25 min |

**Key Finding:** Linear POD fails catastrophically on parameter interpolation (101% error). Nonlinear VAE encoding improves interpolation 4× (25% error).

## Repository Structure
```
├── data/                   # See Google Drive link below
├── PINN/                   # PINN implementation
│   ├── pinn_cylinder_v2.py # Training script
│   ├── pinn_v2_test.py     # Evaluation on unseen frequency
│   └── pinn_v2_model.pt    # Trained model
│   └── results/            # Result figures
├── pod_neural_ode/         # Neural ODE + POD implementation
│   ├── neural_ode_flow.py
│   ├── test_neural_ode.py
│   ├── visualization_neural_ode.py
│   └── neural_ode_model_v3.pt
│   └── results/            # Result figures
├── vae_neural_ode/         # Neural ODE + VAE implementation
│   ├── vae_neural_ode.py
│   └── vae_neural_ode_model.pt
│   ├── visualize_vae_ode.py
│   └── results/            # Result figures

```

## Data

CFD data files are hosted on Google Drive due to file size limits:

📁 **[Download Data Files (Google Drive)](https://drive.google.com/YOUR_FOLDER_LINK_HERE)**

The folder contains:
- `dataset_f01.npz` - f₀ = 0.1 Hz (training)
- `dataset_f02.npz` - f₀ = 0.2 Hz (test/unseen)
- `dataset_f03.npz` - f₀ = 0.3 Hz (training)

Download and place in `data/` folder before running scripts.

### CFD Simulation Details

Simulations performed using OpenFOAM 12 (pimpleFoam solver):

- Reynolds number: Re = 100
- Domain: [-8D, 12D] × [-6D, 6D]
- Oscillation amplitude: A/D = 0.2
- Training frequencies: f₀ = 0.1, 0.3 Hz
- Test frequency: f₀ = 0.2 Hz
- Snapshots: 151 per frequency (t = 15-60s, Δt = 0.3s)

## Requirements
```
python >= 3.8
torch >= 2.0
numpy
matplotlib
scikit-learn
```

## Author

Roopa Adepu  
Northeastern University  
ME5510 - Scientific Machine Learning for Mechanical Engineers  
December 2024

## Acknowledgments

Course Instructor: Dr. Juner Zhu
