"""
Test Neural ODE on Unseen Frequency f0=0.2 Hz (Interpolation Test)
==================================================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import re

# Import directly from your training script!
from neural_ode_flow import ODEFunc, rollout, reconstruct_velocity, compute_errors

# ============================================================
# CONFIG
# ============================================================

CASE_PATH = "./Re_100_f0_02"
MODEL_PATH = "neural_ode_model_v3.pt"
CELL_CENTERS_FILE = "./Re_100_f0_01/2/C"

F0_TEST = 0.2
N_CELLS = 3456

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_U_file(U_path, nCells):
    ux = np.zeros(nCells)
    uy = np.zeros(nCells)
    uz = np.zeros(nCells)

    with open(U_path, "r") as f:
        lines = f.readlines()

    data_started = False
    k = 0

    for line in lines:
        line = line.strip()
        if line == "(":
            data_started = True
            continue
        if line == ");" or line == ")":
            data_started = False
            continue
        if not data_started:
            continue
        if line.startswith("(") and line.endswith(")"):
            inside = line[1:-1].strip()
            vals = inside.split()
            if len(vals) != 3:
                continue
            ux[k] = float(vals[0])
            uy[k] = float(vals[1])
            uz[k] = float(vals[2])
            k += 1

    return np.vstack([ux, uy, uz]).T


def load_openfoam_case(case_path, nCells=3456, t_start=15.0, t_end=60.0, downsample=3):
    dirs = []
    for d in os.listdir(case_path):
        try:
            val = float(d)
            dirs.append(val)
        except:
            continue

    dirs = sorted(dirs)
    dirs = [t for t in dirs if t_start <= t <= t_end]
    dirs = dirs[::downsample]

    print(f"Selected {len(dirs)} snapshots from {case_path}")

    traj = []
    for t in dirs:
        folder = f"{t:.1f}".rstrip("0").rstrip(".")
        U_path = os.path.join(case_path, folder, "U")
        if not os.path.isfile(U_path):
            folder = f"{t:.1f}"
            U_path = os.path.join(case_path, folder, "U")
        u_vec = read_U_file(U_path, nCells=nCells)
        traj.append(u_vec)

    traj = np.array(traj)
    t_array = np.array(dirs)
    return t_array, traj


def parse_cell_centers(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    match = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', content)
    n_cells = int(match.group(1))
    pattern = r'\(([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\)'
    
    start_idx = match.end()
    internal_section = content[start_idx:]
    
    boundary_match = re.search(r'\)\s*;\s*\n\s*boundaryField', internal_section)
    if boundary_match:
        internal_section = internal_section[:boundary_match.start()]
    
    matches = re.findall(pattern, internal_section)
    coords = np.array([[float(x), float(y), float(z)] for x, y, z in matches[:n_cells]])
    
    return coords


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("Neural ODE Interpolation Test: f0 = 0.2 Hz")
    print("="*60)
    
    # --- Load OpenFOAM data ---
    print("\n--- Loading OpenFOAM Data ---")
    t_array, traj_raw = load_openfoam_case(CASE_PATH)
    traj_test = traj_raw[:, :, :2]  # ux, uy only
    n_times = len(t_array)
    dt = float(t_array[1] - t_array[0])
    
    print(f"Trajectory shape: {traj_test.shape}")
    print(f"Time: {t_array[0]} to {t_array[-1]}, dt = {dt}")
    
    # --- Load trained model ---
    print("\n--- Loading Neural ODE Model ---")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    U_mean = checkpoint['U_mean']
    Phi = checkpoint['Phi']
    a_mean = checkpoint['a_mean']
    a_std = checkpoint['a_std']
    f0_mean = checkpoint['f0_mean']
    f0_std = checkpoint['f0_std']
    n_modes = checkpoint['n_modes']
    original_shape = checkpoint['original_shape']
    
    print(f"Loaded: U_mean {U_mean.shape}, Phi {Phi.shape}")
    print(f"f0 normalization: mean={f0_mean}, std={f0_std}")
    
    # Create model using the SAME class from training
    ode_func = ODEFunc(n_modes, hidden_dim=128)
    ode_func.load_state_dict(checkpoint['model_state'])
    ode_func.to(device)
    ode_func.eval()
    print("Model loaded!")
    
    # --- Project test data onto POD basis ---
    print("\n--- Projecting onto POD Basis ---")
    traj_flat = traj_test.reshape(n_times, -1)
    U_mean_flat = U_mean.reshape(-1)
    
    U_centered = traj_flat - U_mean_flat
    a_test = U_centered @ Phi
    a_test_norm = (a_test - a_mean) / a_std
    
    # Normalize f0
    f0_norm = (F0_TEST - f0_mean) / f0_std
    print(f"f0 = {F0_TEST} -> normalized: {f0_norm:.2f}")
    
    # --- Rollout ---
    print("\n--- Rolling out Neural ODE ---")
    a_pred_norm = rollout(ode_func, a_test_norm[0], t_array, dt, f0_norm)
    
    # Denormalize and reconstruct
    a_pred = a_pred_norm * a_std + a_mean
    a_test_denorm = a_test_norm * a_std + a_mean
    U_pred = reconstruct_velocity(a_pred, U_mean, Phi, original_shape)
    
    # --- Compute Errors ---
    print("\n--- Results ---")
    errors = compute_errors(traj_test, U_pred)
    
    print(f"\n{'='*50}")
    print(f"f0 = {F0_TEST} Hz (INTERPOLATION - UNSEEN)")
    print(f"{'='*50}")
    print(f"Mean Relative Error: {errors['mean_rel_error']*100:.2f}%")
    print(f"Max Relative Error:  {errors['max_rel_error']*100:.2f}%")
    print(f"R² Score:            {errors['r2']:.4f}")
    print(f"{'='*50}")
    
    # --- Visualizations ---
    print("\n--- Generating Plots ---")
    cell_centers = parse_cell_centers(CELL_CENTERS_FILE)
    x_cells, y_cells = cell_centers[:, 0], cell_centers[:, 1]
    
    # 1. POD Coefficients
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(6):
        ax = axes[i//3, i%3]
        ax.plot(t_array, a_test_denorm[:, i], 'b-', label='CFD', lw=2)
        ax.plot(t_array, a_pred[:, i], 'r--', label='Neural ODE', lw=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'$a_{{{i+1}}}$')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'POD Coefficients - f₀ = {F0_TEST} Hz (Unseen)', fontsize=14)
    plt.tight_layout()
    plt.savefig('neural_ode_test_f02_coefficients.png', dpi=150)
    print("Saved: neural_ode_test_f02_coefficients.png")
    plt.show()
    
    # 2. Velocity Evolution
    time_indices = [0, n_times//4, n_times//2, 3*n_times//4, n_times-1]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    
    for col, tidx in enumerate(time_indices):
        U_mag_true = np.sqrt(traj_test[tidx,:,0]**2 + traj_test[tidx,:,1]**2)
        U_mag_pred = np.sqrt(U_pred[tidx,:,0]**2 + U_pred[tidx,:,1]**2)
        vmin, vmax = U_mag_true.min(), U_mag_true.max()
        
        axes[0,col].scatter(x_cells, y_cells, c=U_mag_true, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
        axes[0,col].set_title(f't={t_array[tidx]:.1f}s')
        axes[0,col].set_aspect('equal')
        if col == 0: axes[0,col].set_ylabel('CFD')
        
        axes[1,col].scatter(x_cells, y_cells, c=U_mag_pred, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
        axes[1,col].set_aspect('equal')
        if col == 0: axes[1,col].set_ylabel('Neural ODE')
    
    plt.suptitle(f'Velocity Evolution - f₀ = {F0_TEST} Hz (UNSEEN) - Error: {errors["mean_rel_error"]*100:.2f}%', fontsize=14)
    plt.tight_layout()
    plt.savefig('neural_ode_test_f02_evolution.png', dpi=150)
    print("Saved: neural_ode_test_f02_evolution.png")
    plt.show()
    
    # 3. Error Analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(t_array, errors['rel_errors']*100, 'b-', lw=2)
    axes[0].axhline(errors['mean_rel_error']*100, color='r', ls='--', label=f'Mean: {errors["mean_rel_error"]*100:.2f}%')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Relative Error (%)')
    axes[0].set_title('Error Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    U_mag_true = np.sqrt(traj_test[:,:,0]**2 + traj_test[:,:,1]**2)
    U_mag_pred = np.sqrt(U_pred[:,:,0]**2 + U_pred[:,:,1]**2)
    error_spatial = np.mean(np.abs(U_mag_pred - U_mag_true), axis=0)
    sc = axes[1].scatter(x_cells, y_cells, c=error_spatial, cmap='Reds', s=5)
    plt.colorbar(sc, ax=axes[1], label='MAE')
    axes[1].set_title('Spatial Error Distribution')
    axes[1].set_aspect('equal')
    
    cell_idx = 1000
    axes[2].plot(t_array, traj_test[:,cell_idx,0], 'b-', label='CFD', lw=2)
    axes[2].plot(t_array, U_pred[:,cell_idx,0], 'r--', label='Neural ODE', lw=2)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('$u_x$')
    axes[2].set_title(f'Velocity at Cell {cell_idx}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Error Analysis - f₀ = {F0_TEST} Hz', fontsize=14)
    plt.tight_layout()
    plt.savefig('neural_ode_test_f02_error.png', dpi=150)
    print("Saved: neural_ode_test_f02_error.png")
    plt.show()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    
    return errors


if __name__ == "__main__":
    results = main()