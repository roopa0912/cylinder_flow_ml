"""
Standalone Visualization Script for Neural ODE Results
=======================================================
Run this separately after training to generate publication-quality figures.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import re

# ============================================================
# CONFIG - Update these paths as needed
# ============================================================

DATA_F01 = "dataset_f01.npz"
DATA_F03 = "dataset_f03.npz"
MODEL_PATH = "neural_ode_model_v3.pt"
CELL_CENTERS_FILE = "Re_100_f0_01/2/C"  # Path to OpenFOAM C file

N_MODES = 15
HIDDEN_DIM = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. PARSE OPENFOAM CELL CENTERS
# ============================================================

def parse_openfoam_cell_centers(filepath):
    """
    Parse OpenFOAM C (cell centers) file.
    Returns numpy array of shape (n_cells, 3) with x, y, z coordinates.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the internalField section
    # Look for pattern: internalField nonuniform List<vector> \n N \n (
    match = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', content)
    if not match:
        raise ValueError("Could not find internalField in file")
    
    n_cells = int(match.group(1))
    print(f"Found {n_cells} cells")
    
    # Find all coordinate tuples (x y z)
    # Pattern matches (number number number) allowing for scientific notation
    pattern = r'\(([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\)'
    
    # Get content after internalField declaration
    start_idx = match.end()
    internal_section = content[start_idx:]
    
    # Find the closing parenthesis of internalField (before boundaryField)
    boundary_match = re.search(r'\)\s*;\s*\n\s*boundaryField', internal_section)
    if boundary_match:
        internal_section = internal_section[:boundary_match.start()]
    
    matches = re.findall(pattern, internal_section)
    
    if len(matches) < n_cells:
        print(f"Warning: Found {len(matches)} coordinates, expected {n_cells}")
    
    # Take only the first n_cells matches (internal field)
    coords = np.array([[float(x), float(y), float(z)] for x, y, z in matches[:n_cells]])
    
    print(f"Parsed cell centers shape: {coords.shape}")
    return coords


# ============================================================
# 2. NEURAL ODE MODEL (same as training)
# ============================================================

class ODEFunc(nn.Module):
    def __init__(self, n_modes, hidden_dim=128):
        super().__init__()
        self.n_modes = n_modes
        self.net = nn.Sequential(
            nn.Linear(n_modes + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_modes),
        )
    
    def forward(self, a, f0):
        inp = torch.cat([a, f0], dim=-1)
        return self.net(inp)


# ============================================================
# 3. POD AND ROLLOUT FUNCTIONS
# ============================================================

def compute_pod(traj1, traj2, n_modes):
    """Compute POD from trajectories."""
    traj1_flat = traj1.reshape(traj1.shape[0], -1)
    traj2_flat = traj2.reshape(traj2.shape[0], -1)
    
    all_snapshots = np.vstack([traj1_flat, traj2_flat])
    U_mean = all_snapshots.mean(axis=0, keepdims=True)
    U_centered = all_snapshots - U_mean
    
    U_svd, S, Vt = np.linalg.svd(U_centered, full_matrices=False)
    Phi = Vt[:n_modes, :].T
    
    a_all = U_centered @ Phi
    T1 = traj1.shape[0]
    a1 = a_all[:T1, :]
    a2 = a_all[T1:, :]
    
    return U_mean, Phi, a1, a2


def reconstruct_velocity(a, U_mean, Phi, original_shape):
    """Reconstruct velocity field from POD coefficients."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    U_flat = U_mean + a @ Phi.T
    U = U_flat.reshape(-1, *original_shape)
    return U


def rollout(ode_func, a0, n_steps, dt, f0):
    """Integrate Neural ODE."""
    ode_func.eval()
    a = torch.from_numpy(a0.astype(np.float32)).unsqueeze(0).to(device)
    f0_tensor = torch.tensor([[f0]], dtype=torch.float32).to(device)
    trajectory = [a0]
    
    with torch.no_grad():
        for _ in range(n_steps):
            k1 = ode_func(a, f0_tensor)
            k2 = ode_func(a + 0.5 * dt * k1, f0_tensor)
            k3 = ode_func(a + 0.5 * dt * k2, f0_tensor)
            k4 = ode_func(a + dt * k3, f0_tensor)
            a = a + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(a.cpu().numpy()[0])
    
    return np.array(trajectory)


# ============================================================
# 4. VISUALIZATION FUNCTIONS
# ============================================================

def plot_velocity_field_comparison(U_true, U_pred, t_array, time_idx, cell_xy, f0_val, save_name):
    """Plot CFD vs Neural ODE velocity magnitude at a specific time."""
    
    t_val = t_array[time_idx]
    
    U_mag_true = np.sqrt(U_true[time_idx, :, 0]**2 + U_true[time_idx, :, 1]**2)
    U_mag_pred = np.sqrt(U_pred[time_idx, :, 0]**2 + U_pred[time_idx, :, 1]**2)
    error = np.abs(U_mag_pred - U_mag_true)
    
    vmin, vmax = U_mag_true.min(), U_mag_true.max()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # CFD
    sc1 = axes[0].scatter(cell_xy[:, 0], cell_xy[:, 1], 
                          c=U_mag_true, cmap='coolwarm', s=8, vmin=vmin, vmax=vmax)
    axes[0].set_title('CFD Ground Truth', fontsize=12)
    axes[0].set_xlabel('x/D')
    axes[0].set_ylabel('y/D')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], label='|U|')
    
    # Neural ODE
    sc2 = axes[1].scatter(cell_xy[:, 0], cell_xy[:, 1], 
                          c=U_mag_pred, cmap='coolwarm', s=8, vmin=vmin, vmax=vmax)
    axes[1].set_title('Neural ODE Prediction', fontsize=12)
    axes[1].set_xlabel('x/D')
    axes[1].set_ylabel('y/D')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], label='|U|')
    
    # Error
    sc3 = axes[2].scatter(cell_xy[:, 0], cell_xy[:, 1], 
                          c=error, cmap='Reds', s=8)
    axes[2].set_title('Absolute Error', fontsize=12)
    axes[2].set_xlabel('x/D')
    axes[2].set_ylabel('y/D')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2], label='|Error|')
    
    plt.suptitle(f'Velocity Field: f₀={f0_val} Hz, t={t_val:.1f}s', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_name}")
    plt.close()


def plot_multi_timestep_comparison(U_true, U_pred, t_array, cell_xy, f0_val, save_name):
    """Plot velocity comparison at multiple time steps."""
    
    # Select 5 time indices spread across the simulation
    n_times = len(t_array)
    time_indices = [0, n_times//4, n_times//2, 3*n_times//4, n_times-1]
    
    fig, axes = plt.subplots(2, len(time_indices), figsize=(18, 7))
    
    for col, tidx in enumerate(time_indices):
        t_val = t_array[tidx]
        
        U_mag_true = np.sqrt(U_true[tidx, :, 0]**2 + U_true[tidx, :, 1]**2)
        U_mag_pred = np.sqrt(U_pred[tidx, :, 0]**2 + U_pred[tidx, :, 1]**2)
        
        vmin, vmax = U_mag_true.min(), U_mag_true.max()
        
        # CFD (top row)
        sc1 = axes[0, col].scatter(cell_xy[:, 0], cell_xy[:, 1], 
                                    c=U_mag_true, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f't = {t_val:.1f}s', fontsize=11)
        axes[0, col].set_aspect('equal')
        axes[0, col].set_xlim([-8, 12])
        axes[0, col].set_ylim([-6, 6])
        if col == 0:
            axes[0, col].set_ylabel('CFD', fontsize=12, fontweight='bold')
        
        # Neural ODE (bottom row)
        sc2 = axes[1, col].scatter(cell_xy[:, 0], cell_xy[:, 1], 
                                    c=U_mag_pred, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
        axes[1, col].set_aspect('equal')
        axes[1, col].set_xlim([-8, 12])
        axes[1, col].set_ylim([-6, 6])
        if col == 0:
            axes[1, col].set_ylabel('Neural ODE', fontsize=12, fontweight='bold')
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sc1, cax=cbar_ax, label='|U| (m/s)')
    
    plt.suptitle(f'Velocity Magnitude Evolution (f₀ = {f0_val} Hz)', fontsize=14, fontweight='bold')
    plt.savefig(save_name, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_name}")
    plt.close()


def plot_wake_profile(U_true, U_pred, t_array, time_idx, cell_xy, f0_val, save_name):
    """Plot velocity profiles in the wake region at different x locations."""
    
    t_val = t_array[time_idx]
    
    # Select x locations in the wake
    x_locations = [1, 3, 6, 9]
    
    fig, axes = plt.subplots(1, len(x_locations), figsize=(16, 4))
    
    for i, x_loc in enumerate(x_locations):
        # Find cells near this x location
        tol = 0.5
        mask = np.abs(cell_xy[:, 0] - x_loc) < tol
        
        if np.sum(mask) > 0:
            y_vals = cell_xy[mask, 1]
            u_true = U_true[time_idx, mask, 0]
            u_pred = U_pred[time_idx, mask, 0]
            
            # Sort by y
            sort_idx = np.argsort(y_vals)
            
            axes[i].plot(u_true[sort_idx], y_vals[sort_idx], 'b-', label='CFD', linewidth=2)
            axes[i].plot(u_pred[sort_idx], y_vals[sort_idx], 'r--', label='Neural ODE', linewidth=2)
            axes[i].set_xlabel('u_x (m/s)')
            axes[i].set_ylabel('y/D')
            axes[i].set_title(f'x/D = {x_loc}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Wake Velocity Profiles (f₀ = {f0_val} Hz, t = {t_val:.1f}s)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_name}")
    plt.close()


def plot_error_distribution(U_true, U_pred, cell_xy, save_name):
    """Plot spatial distribution of time-averaged error."""
    
    # Compute time-averaged error at each cell
    U_mag_true = np.sqrt(U_true[:, :, 0]**2 + U_true[:, :, 1]**2)
    U_mag_pred = np.sqrt(U_pred[:, :, 0]**2 + U_pred[:, :, 1]**2)
    
    # Mean absolute error over time
    mae = np.mean(np.abs(U_mag_pred - U_mag_true), axis=0)
    
    # Mean relative error over time
    mre = np.mean(np.abs(U_mag_pred - U_mag_true) / (U_mag_true + 1e-10), axis=0) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sc1 = axes[0].scatter(cell_xy[:, 0], cell_xy[:, 1], c=mae, cmap='hot', s=8)
    axes[0].set_title('Mean Absolute Error')
    axes[0].set_xlabel('x/D')
    axes[0].set_ylabel('y/D')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], label='MAE')
    
    sc2 = axes[1].scatter(cell_xy[:, 0], cell_xy[:, 1], c=mre, cmap='hot', s=8, vmax=20)
    axes[1].set_title('Mean Relative Error (%)')
    axes[1].set_xlabel('x/D')
    axes[1].set_ylabel('y/D')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], label='MRE (%)')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_name}")
    plt.close()


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Neural ODE Visualization")
    print("=" * 60)
    
    # Load cell centers
    print("\n--- Loading Cell Centers ---")
    cell_centers = parse_openfoam_cell_centers(CELL_CENTERS_FILE)
    cell_xy = cell_centers[:, :2]  # Only x, y
    
    # Load data
    print("\n--- Loading Data ---")
    d1 = np.load(DATA_F01)
    d2 = np.load(DATA_F03)
    t1, traj1 = d1['t'], d1['traj'][:, :, :2]
    t2, traj2 = d2['t'], d2['traj'][:, :, :2]
    dt = float(t1[1] - t1[0])
    print(f"Loaded: {traj1.shape}, {traj2.shape}")
    
    original_shape = traj1.shape[1:]
    
    # Compute POD
    print("\n--- Computing POD ---")
    U_mean, Phi, a1, a2 = compute_pod(traj1, traj2, N_MODES)
    
    # Normalize
    a_all = np.vstack([a1, a2])
    a_mean = a_all.mean(axis=0)
    a_std = a_all.std(axis=0) + 1e-8
    a1_norm = (a1 - a_mean) / a_std
    a2_norm = (a2 - a_mean) / a_std
    
    F0_1, F0_2 = 0.1, 0.3
    f0_mean = (F0_1 + F0_2) / 2
    f0_std = (F0_2 - F0_1) / 2 + 1e-8
    f0_1_norm = (F0_1 - f0_mean) / f0_std
    f0_2_norm = (F0_2 - f0_mean) / f0_std
    
    # Load model
    print("\n--- Loading Model ---")
    ode_func = ODEFunc(N_MODES, HIDDEN_DIM)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    ode_func.load_state_dict(checkpoint['model_state'])
    ode_func.to(device)
    ode_func.eval()
    print("Model loaded successfully!")
    
    # Rollout
    print("\n--- Running Rollout ---")
    n_steps = len(t1) - 1
    
    # f0 = 0.1
    a_pred_norm = rollout(ode_func, a1_norm[0], n_steps, dt, f0_1_norm)
    a_pred = a_pred_norm * a_std + a_mean
    U_pred = reconstruct_velocity(a_pred, U_mean, Phi, original_shape)
    
    # f0 = 0.3
    a_pred_norm_2 = rollout(ode_func, a2_norm[0], n_steps, dt, f0_2_norm)
    a_pred_2 = a_pred_norm_2 * a_std + a_mean
    U_pred_2 = reconstruct_velocity(a_pred_2, U_mean, Phi, original_shape)
    
    # Generate visualizations
    print("\n--- Generating Visualizations ---")
    
    # 1. Multi-timestep comparison
    plot_multi_timestep_comparison(traj1, U_pred, t1, cell_xy, 0.1, 'viz_evolution_f01.png')
    plot_multi_timestep_comparison(traj2, U_pred_2, t2, cell_xy, 0.3, 'viz_evolution_f03.png')
    
    # 2. Detailed comparison at specific times
    for tidx in [0, 50, 100]:
        plot_velocity_field_comparison(traj1, U_pred, t1, tidx, cell_xy, 0.1, f'viz_detail_f01_t{tidx}.png')
    
    # 3. Wake profiles
    plot_wake_profile(traj1, U_pred, t1, 75, cell_xy, 0.1, 'viz_wake_f01.png')
    plot_wake_profile(traj2, U_pred_2, t2, 75, cell_xy, 0.3, 'viz_wake_f03.png')
    
    # 4. Error distribution
    plot_error_distribution(traj1, U_pred, cell_xy, 'viz_error_dist_f01.png')
    plot_error_distribution(traj2, U_pred_2, cell_xy, 'viz_error_dist_f03.png')
    
    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()