"""
Visualize VAE + Neural ODE Results
==================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import re

# Import from your VAE training script
from vae_neural_ode import FlowVAE, LatentODEFunc, rollout_vae_ode, compute_errors

# ============================================================
# CONFIG
# ============================================================

DATA_F01 = "dataset_f01.npz"
DATA_F03 = "dataset_f03.npz"
DATA_F02 = "dataset_f02.npz"  # Test data!
MODEL_PATH = "vae_neural_ode_model.pt"
CELL_CENTERS_FILE = "Re_100_f0_01/2/C"

F0_1, F0_2, F0_TEST = 0.1, 0.3, 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# HELPERS
# ============================================================

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


def load_data():
    """Load all datasets."""
    d1 = np.load(DATA_F01)
    d2 = np.load(DATA_F03)
    d3 = np.load(DATA_F02)
    
    t = d1['t']
    traj1 = d1['traj'][:, :, :2] * d1['std'] + d1['mean']
    traj2 = d2['traj'][:, :, :2] * d2['std'] + d2['mean']
    traj3 = d3['traj'][:, :, :2]  # f0=0.2 (already unnormalized in test script)
    
    # Flatten
    traj1_flat = traj1.reshape(traj1.shape[0], -1)
    traj2_flat = traj2.reshape(traj2.shape[0], -1)
    traj3_flat = traj3.reshape(traj3.shape[0], -1)
    
    dt = float(t[1] - t[0])
    
    return t, traj1_flat, traj2_flat, traj3_flat, traj1.shape[1:], dt


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("VAE + Neural ODE Visualization")
    print("="*60)
    
    # Load data
    print("\n--- Loading Data ---")
    t_array, traj1, traj2, traj3, original_shape, dt = load_data()
    n_times = len(t_array)
    n_cells = original_shape[0]
    
    # Load model
    print("\n--- Loading Model ---")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    data_mean = checkpoint['data_mean']
    data_std = checkpoint['data_std']
    f0_mean = checkpoint['f0_mean']
    f0_std = checkpoint['f0_std']
    latent_dim = checkpoint['latent_dim']
    
    vae = FlowVAE(input_dim=6912, latent_dim=latent_dim, hidden_dim=256)
    vae.load_state_dict(checkpoint['vae_state'])
    vae.to(device)
    vae.eval()
    
    ode_func = LatentODEFunc(latent_dim=latent_dim, hidden_dim=128)
    ode_func.load_state_dict(checkpoint['ode_state'])
    ode_func.to(device)
    ode_func.eval()
    
    print(f"Loaded VAE + ODE model (latent_dim={latent_dim})")
    
    # Normalize data
    traj1_norm = (traj1 - data_mean) / data_std
    traj2_norm = (traj2 - data_mean) / data_std
    traj3_norm = (traj3 - data_mean) / data_std
    
    # Get initial latent codes
    with torch.no_grad():
        z1_all = vae.get_latent(torch.tensor(traj1_norm, dtype=torch.float32).to(device)).cpu().numpy()
        z2_all = vae.get_latent(torch.tensor(traj2_norm, dtype=torch.float32).to(device)).cpu().numpy()
        z3_all = vae.get_latent(torch.tensor(traj3_norm, dtype=torch.float32).to(device)).cpu().numpy()
    
    # Normalized frequencies
    f0_1_norm = (F0_1 - f0_mean) / f0_std
    f0_2_norm = (F0_2 - f0_mean) / f0_std
    f0_test_norm = (F0_TEST - f0_mean) / f0_std
    
    print(f"f0 normalization: {F0_1}->{f0_1_norm:.2f}, {F0_2}->{f0_2_norm:.2f}, {F0_TEST}->{f0_test_norm:.2f}")
    
    # Rollout for all frequencies
    print("\n--- Rolling out predictions ---")
    
    z_pred_1, U_pred_norm_1 = rollout_vae_ode(vae, ode_func, z1_all[0], t_array, dt, f0_1_norm)
    z_pred_2, U_pred_norm_2 = rollout_vae_ode(vae, ode_func, z2_all[0], t_array, dt, f0_2_norm)
    z_pred_3, U_pred_norm_3 = rollout_vae_ode(vae, ode_func, z3_all[0], t_array, dt, f0_test_norm)
    
    # Denormalize
    U_pred_1 = U_pred_norm_1 * data_std + data_mean
    U_pred_2 = U_pred_norm_2 * data_std + data_mean
    U_pred_3 = U_pred_norm_3 * data_std + data_mean
    
    # Compute errors
    errors_1 = compute_errors(traj1, U_pred_1)
    errors_2 = compute_errors(traj2, U_pred_2)
    errors_3 = compute_errors(traj3, U_pred_3)
    
    print(f"\n{'='*60}")
    print("VAE + Neural ODE Results Summary")
    print(f"{'='*60}")
    print(f"f0=0.1 Hz (train): {errors_1['mean_rel_error']*100:.2f}% error, R²={errors_1['r2']:.4f}")
    print(f"f0=0.3 Hz (train): {errors_2['mean_rel_error']*100:.2f}% error, R²={errors_2['r2']:.4f}")
    print(f"f0=0.2 Hz (TEST):  {errors_3['mean_rel_error']*100:.2f}% error, R²={errors_3['r2']:.4f}")
    print(f"{'='*60}")
    
    # Load cell centers for plotting
    cell_centers = parse_cell_centers(CELL_CENTERS_FILE)
    x_cells, y_cells = cell_centers[:, 0], cell_centers[:, 1]
    
    # ============================================================
    # PLOT 1: Velocity Evolution for all three frequencies
    # ============================================================
    print("\n--- Generating Visualizations ---")
    
    for f0, traj_true, U_pred, errors, label in [
        (F0_1, traj1, U_pred_1, errors_1, 'f01'),
        (F0_2, traj2, U_pred_2, errors_2, 'f03'),
        (F0_TEST, traj3, U_pred_3, errors_3, 'f02_TEST'),
    ]:
        # Reshape for plotting
        U_true_2d = traj_true.reshape(n_times, n_cells, 2)
        U_pred_2d = U_pred.reshape(n_times, n_cells, 2)
        
        time_indices = [0, n_times//4, n_times//2, 3*n_times//4, n_times-1]
        
        fig, axes = plt.subplots(3, len(time_indices), figsize=(18, 10))
        
        for col, tidx in enumerate(time_indices):
            t_val = t_array[tidx]
            
            U_mag_true = np.sqrt(U_true_2d[tidx,:,0]**2 + U_true_2d[tidx,:,1]**2)
            U_mag_pred = np.sqrt(U_pred_2d[tidx,:,0]**2 + U_pred_2d[tidx,:,1]**2)
            error_field = np.abs(U_mag_pred - U_mag_true)
            
            vmin, vmax = U_mag_true.min(), U_mag_true.max()
            
            axes[0,col].scatter(x_cells, y_cells, c=U_mag_true, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
            axes[0,col].set_title(f't = {t_val:.1f}s')
            axes[0,col].set_aspect('equal')
            if col == 0: axes[0,col].set_ylabel('CFD', fontsize=12)
            
            axes[1,col].scatter(x_cells, y_cells, c=U_mag_pred, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
            axes[1,col].set_aspect('equal')
            if col == 0: axes[1,col].set_ylabel('VAE+ODE', fontsize=12)
            
            axes[2,col].scatter(x_cells, y_cells, c=error_field, cmap='Reds', s=3)
            axes[2,col].set_aspect('equal')
            if col == 0: axes[2,col].set_ylabel('Error', fontsize=12)
        
        test_label = " (UNSEEN)" if f0 == F0_TEST else " (train)"
        plt.suptitle(f'VAE+Neural ODE: f₀={f0} Hz{test_label} - Mean Error: {errors["mean_rel_error"]*100:.2f}%', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'vae_ode_evolution_{label}.png', dpi=150)
        print(f"Saved: vae_ode_evolution_{label}.png")
        plt.show()
    
    # ============================================================
    # PLOT 2: Latent Space Trajectories Comparison
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i in range(6):
        ax = axes[i//3, i%3]
        ax.plot(t_array, z1_all[:, i], 'b-', label='True', lw=2, alpha=0.7)
        ax.plot(t_array, z_pred_1[:, i], 'r--', label='Pred', lw=2, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'$z_{{{i+1}}}$')
        ax.set_title(f'Latent Dim {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Latent Trajectories (f₀=0.1 Hz)', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_ode_latent_f01.png', dpi=150)
    print("Saved: vae_ode_latent_f01.png")
    plt.show()
    
    # ============================================================
    # PLOT 3: Latent Space for TEST frequency (f0=0.2)
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i in range(6):
        ax = axes[i//3, i%3]
        ax.plot(t_array, z3_all[:, i], 'b-', label='True (encoded)', lw=2, alpha=0.7)
        ax.plot(t_array, z_pred_3[:, i], 'r--', label='Predicted', lw=2, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'$z_{{{i+1}}}$')
        ax.set_title(f'Latent Dim {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Latent Trajectories f₀=0.2 Hz (UNSEEN FREQUENCY)', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_ode_latent_f02_TEST.png', dpi=150)
    print("Saved: vae_ode_latent_f02_TEST.png")
    plt.show()
    
    # ============================================================
    # PLOT 4: Error Analysis - All Frequencies
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    ax.plot(t_array, errors_1['rel_errors']*100, 'b-', label='f₀=0.1 (train)', lw=2)
    ax.plot(t_array, errors_2['rel_errors']*100, 'g-', label='f₀=0.3 (train)', lw=2)
    ax.plot(t_array, errors_3['rel_errors']*100, 'r-', label='f₀=0.2 (TEST)', lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Error Over Time - All Frequencies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    U_true_2d = traj3.reshape(n_times, n_cells, 2)
    U_pred_2d = U_pred_3.reshape(n_times, n_cells, 2)
    U_mag_true = np.sqrt(U_true_2d[:,:,0]**2 + U_true_2d[:,:,1]**2)
    U_mag_pred = np.sqrt(U_pred_2d[:,:,0]**2 + U_pred_2d[:,:,1]**2)
    error_spatial = np.mean(np.abs(U_mag_pred - U_mag_true), axis=0)
    sc = ax.scatter(x_cells, y_cells, c=error_spatial, cmap='Reds', s=5)
    plt.colorbar(sc, ax=ax, label='MAE')
    ax.set_title('Spatial Error Distribution (f₀=0.2 TEST)')
    ax.set_aspect('equal')
    
    ax = axes[2]
    methods = ['f₀=0.1\n(train)', 'f₀=0.3\n(train)', 'f₀=0.2\n(TEST)']
    errors_list = [errors_1['mean_rel_error']*100, errors_2['mean_rel_error']*100, errors_3['mean_rel_error']*100]
    colors = ['blue', 'green', 'red']
    bars = ax.bar(methods, errors_list, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Relative Error (%)')
    ax.set_title('Error Comparison')
    for bar, err in zip(bars, errors_list):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{err:.2f}%', 
                ha='center', va='bottom', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('vae_ode_error_analysis.png', dpi=150)
    print("Saved: vae_ode_error_analysis.png")
    plt.show()
    
    # ============================================================
    # PLOT 5: Detailed Comparison at Mid-Time for TEST frequency
    # ============================================================
    mid_idx = n_times // 2
    t_mid = t_array[mid_idx]
    
    U_true_2d = traj3.reshape(n_times, n_cells, 2)
    U_pred_2d = U_pred_3.reshape(n_times, n_cells, 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    U_mag_true = np.sqrt(U_true_2d[mid_idx,:,0]**2 + U_true_2d[mid_idx,:,1]**2)
    U_mag_pred = np.sqrt(U_pred_2d[mid_idx,:,0]**2 + U_pred_2d[mid_idx,:,1]**2)
    error_field = np.abs(U_mag_pred - U_mag_true)
    
    vmin, vmax = U_mag_true.min(), U_mag_true.max()
    
    sc1 = axes[0].scatter(x_cells, y_cells, c=U_mag_true, cmap='coolwarm', s=5, vmin=vmin, vmax=vmax)
    axes[0].set_title(f'CFD Ground Truth (t={t_mid:.1f}s)')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], label='|U|')
    
    sc2 = axes[1].scatter(x_cells, y_cells, c=U_mag_pred, cmap='coolwarm', s=5, vmin=vmin, vmax=vmax)
    axes[1].set_title('VAE+ODE Prediction')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], label='|U|')
    
    sc3 = axes[2].scatter(x_cells, y_cells, c=error_field, cmap='Reds', s=5)
    axes[2].set_title('Absolute Error')
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2], label='|Error|')
    
    plt.suptitle(f'VAE+Neural ODE: f₀=0.2 Hz (UNSEEN) - Error: {errors_3["mean_rel_error"]*100:.2f}%', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_ode_detail_f02_TEST.png', dpi=150)
    print("Saved: vae_ode_detail_f02_TEST.png")
    plt.show()
    
    # ============================================================
    # PLOT 6: Method Comparison - POD vs VAE
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['POD+ODE\nf₀=0.1', 'POD+ODE\nf₀=0.3', 'POD+ODE\nf₀=0.2', 
               'VAE+ODE\nf₀=0.1', 'VAE+ODE\nf₀=0.3', 'VAE+ODE\nf₀=0.2']
    errors = [5.65, 4.77, 101.73,  # POD results
              errors_1['mean_rel_error']*100, errors_2['mean_rel_error']*100, errors_3['mean_rel_error']*100]
    colors = ['lightblue', 'lightblue', 'salmon', 'blue', 'blue', 'red']
    
    bars = ax.bar(methods, errors, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('Mean Relative Error (%)', fontsize=12)
    ax.set_title('POD+Neural ODE vs VAE+Neural ODE Comparison', fontsize=14)
    
    # Add value labels
    for bar, err in zip(bars, errors):
        if err < 20:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{err:.1f}%', 
                    ha='center', va='bottom', fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 10, f'{err:.1f}%', 
                    ha='center', va='top', fontsize=10, color='white', fontweight='bold')
    
    ax.axhline(10, color='green', linestyle='--', label='10% threshold')
    ax.set_ylim(0, 120)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('vae_vs_pod_comparison.png', dpi=150)
    print("Saved: vae_vs_pod_comparison.png")
    plt.show()
    
    # ============================================================
    # PLOT 7: Velocity Time Series Comparison
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    cell_idx = 1000
    
    for ax, (f0, traj_true, U_pred, label) in zip(axes, [
        (F0_1, traj1, U_pred_1, 'f₀=0.1 (train)'),
        (F0_2, traj2, U_pred_2, 'f₀=0.3 (train)'),
        (F0_TEST, traj3, U_pred_3, 'f₀=0.2 (TEST)'),
    ]):
        ax.plot(t_array, traj_true[:, cell_idx], 'b-', label='CFD', lw=2)
        ax.plot(t_array, U_pred[:, cell_idx], 'r--', label='VAE+ODE', lw=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'{label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Velocity at Cell {cell_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig('vae_ode_velocity_comparison.png', dpi=150)
    print("Saved: vae_ode_velocity_comparison.png")
    plt.show()
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)


if __name__ == "__main__":
    main()