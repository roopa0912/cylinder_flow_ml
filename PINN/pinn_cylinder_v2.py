"""
Physics-Informed Neural Network (PINN) for Oscillating Cylinder Flow
=====================================================================
VERSION 2: More data, prioritize data fitting
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import re
import time

# ============================================================
# CONFIG - MODIFIED FOR BETTER RESULTS
# ============================================================

DATA_F01 = "data/dataset_f01.npz"
DATA_F03 = "data/dataset_f03.npz"
CELL_CENTERS_FILE = "OpenFOAM/Re_100_f0_01/2/C"

# Physics parameters
RE = 100.0
AMPLITUDE = 0.2
F0_1 = 0.1
F0_2 = 0.3

# Sampling parameters - CHANGED
TIME_DOWNSAMPLE = 1           # CHANGED: Every snapshot (was 5)
SPATIAL_SAMPLES = 500         # Keep same
N_COLLOCATION = 5000          # CHANGED: Reduced (was 10000)
N_BC = 1000                   # CHANGED: Reduced (was 2000)

# Training parameters
EPOCHS = 10000
LR = 1e-3
PATIENCE = 1500               # CHANGED: More patience (was 1000)

# Network architecture - CHANGED
HIDDEN_LAYERS = 6
HIDDEN_DIM = 128              # CHANGED: Bigger (was 64)

# Loss weights - CHANGED: Prioritize data fitting
LAMBDA_DATA = 10.0            # CHANGED: Increased (was 1.0)
LAMBDA_PDE = 0.01             # CHANGED: Decreased (was 0.1)
LAMBDA_BC = 0.1               # CHANGED: Decreased (was 1.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. DATA LOADING (unchanged)
# ============================================================

def parse_openfoam_cell_centers(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    match = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', content)
    if not match:
        raise ValueError("Could not find internalField in file")
    
    n_cells = int(match.group(1))
    pattern = r'\(([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\s+([+-]?\d*\.?\d+(?:e[+-]?\d+)?)\)'
    
    start_idx = match.end()
    internal_section = content[start_idx:]
    
    boundary_match = re.search(r'\)\s*;\s*\n\s*boundaryField', internal_section)
    if boundary_match:
        internal_section = internal_section[:boundary_match.start()]
    
    matches = re.findall(pattern, internal_section)
    coords = np.array([[float(x), float(y), float(z)] for x, y, z in matches[:n_cells]])
    
    print(f"Loaded {coords.shape[0]} cell centers")
    return coords


def load_and_prepare_data():
    cell_centers = parse_openfoam_cell_centers(CELL_CENTERS_FILE)
    x_cells = cell_centers[:, 0]
    y_cells = cell_centers[:, 1]
    n_cells = len(x_cells)
    
    data_f01 = np.load(DATA_F01)
    data_f03 = np.load(DATA_F03)
    
    t1 = data_f01['t']
    traj1 = data_f01['traj'] * data_f01['std'] + data_f01['mean']
    
    t2 = data_f03['t']
    traj2 = data_f03['traj'] * data_f03['std'] + data_f03['mean']
    
    print(f"Loaded trajectories: {traj1.shape}, {traj2.shape}")
    print(f"Time range: {t1[0]} to {t1[-1]}")
    print(f"Velocity range: [{traj1[:,:,:2].min():.3f}, {traj1[:,:,:2].max():.3f}]")
    
    time_indices = np.arange(0, len(t1), TIME_DOWNSAMPLE)
    t1_sparse = t1[time_indices]
    t2_sparse = t2[time_indices]
    traj1_sparse = traj1[time_indices]
    traj2_sparse = traj2[time_indices]
    
    print(f"Time sampling: {len(time_indices)} snapshots")
    
    data_points = []
    
    for case_idx, (t_arr, traj, f0) in enumerate([
        (t1_sparse, traj1_sparse, F0_1),
        (t2_sparse, traj2_sparse, F0_2)
    ]):
        for i, t_val in enumerate(t_arr):
            cell_indices = np.random.choice(n_cells, SPATIAL_SAMPLES, replace=False)
            
            for idx in cell_indices:
                x = x_cells[idx]
                y = y_cells[idx]
                u = traj[i, idx, 0]
                v = traj[i, idx, 1]
                
                data_points.append([x, y, t_val, f0, u, v])
    
    data_points = np.array(data_points)
    print(f"Total data points: {data_points.shape[0]}")
    
    X_data = data_points[:, :4]
    Y_data = data_points[:, 4:]
    
    return X_data, Y_data, x_cells, y_cells, t1, traj1, traj2


def generate_collocation_points(x_cells, y_cells, t_min, t_max):
    x_min, x_max = x_cells.min() + 0.5, x_cells.max() - 0.5
    y_min, y_max = y_cells.min() + 0.5, y_cells.max() - 0.5
    
    x_coll = np.random.uniform(x_min, x_max, N_COLLOCATION)
    y_coll = np.random.uniform(y_min, y_max, N_COLLOCATION)
    t_coll = np.random.uniform(t_min, t_max, N_COLLOCATION)
    
    f0_coll = np.random.choice([F0_1, F0_2], N_COLLOCATION)
    
    r = np.sqrt(x_coll**2 + y_coll**2)
    mask = r > 0.7
    
    X_coll = np.stack([x_coll[mask], y_coll[mask], t_coll[mask], f0_coll[mask]], axis=1)
    print(f"Collocation points: {X_coll.shape[0]}")
    
    return X_coll


def generate_bc_points(t_min, t_max):
    bc_points = []
    bc_types = []
    bc_values = []
    
    n_per_bc = N_BC // 6
    
    for f0 in [F0_1, F0_2]:
        y_inlet = np.random.uniform(-6, 6, n_per_bc)
        t_inlet = np.random.uniform(t_min, t_max, n_per_bc)
        for i in range(n_per_bc):
            bc_points.append([-8.0, y_inlet[i], t_inlet[i], f0])
            bc_types.append(0)
            bc_values.append([1.0, 0.0])
        
        y_outlet = np.random.uniform(-6, 6, n_per_bc)
        t_outlet = np.random.uniform(t_min, t_max, n_per_bc)
        for i in range(n_per_bc):
            bc_points.append([12.0, y_outlet[i], t_outlet[i], f0])
            bc_types.append(1)
            bc_values.append([1.0, 0.0])
        
        theta = np.random.uniform(0, 2*np.pi, n_per_bc)
        t_cyl = np.random.uniform(t_min, t_max, n_per_bc)
        for i in range(n_per_bc):
            y_center = AMPLITUDE * np.sin(2 * np.pi * f0 * t_cyl[i])
            v_cylinder = 2 * np.pi * f0 * AMPLITUDE * np.cos(2 * np.pi * f0 * t_cyl[i])
            
            x_cyl = 0.5 * np.cos(theta[i])
            y_cyl = y_center + 0.5 * np.sin(theta[i])
            
            bc_points.append([x_cyl, y_cyl, t_cyl[i], f0])
            bc_types.append(2)
            bc_values.append([0.0, v_cylinder])
    
    bc_points = np.array(bc_points)
    bc_types = np.array(bc_types)
    bc_values = np.array(bc_values)
    
    print(f"BC points: {bc_points.shape[0]}")
    
    return bc_points, bc_types, bc_values


# ============================================================
# 2. PINN MODEL (unchanged structure, but will be bigger due to HIDDEN_DIM=128)
# ============================================================

class PINN(nn.Module):
    def __init__(self, hidden_layers=HIDDEN_LAYERS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(4, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.net = nn.Sequential(*layers)
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, y, t, f0):
        inputs = torch.stack([x, y, t, f0], dim=1)
        outputs = self.net(inputs)
        u = outputs[:, 0]
        v = outputs[:, 1]
        p = outputs[:, 2]
        return u, v, p


# ============================================================
# 3. PHYSICS RESIDUALS (unchanged)
# ============================================================

def compute_pde_residual(model, x, y, t, f0, Re=RE):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)
    
    u, v, p = model(x, y, t, f0)
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    
    continuity = u_x + v_y
    momentum_x = u_t + u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
    momentum_y = v_t + u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)
    
    return continuity, momentum_x, momentum_y


# ============================================================
# 4. LOSS FUNCTIONS (unchanged)
# ============================================================

def data_loss(model, X_data, Y_data):
    x = X_data[:, 0]
    y = X_data[:, 1]
    t = X_data[:, 2]
    f0 = X_data[:, 3]
    
    u_true = Y_data[:, 0]
    v_true = Y_data[:, 1]
    
    u_pred, v_pred, _ = model(x, y, t, f0)
    
    loss_u = torch.mean((u_pred - u_true)**2)
    loss_v = torch.mean((v_pred - v_true)**2)
    
    return loss_u + loss_v


def pde_loss(model, X_coll):
    x = X_coll[:, 0]
    y = X_coll[:, 1]
    t = X_coll[:, 2]
    f0 = X_coll[:, 3]
    
    r_cont, r_mom_x, r_mom_y = compute_pde_residual(model, x, y, t, f0)
    
    loss = torch.mean(r_cont**2) + torch.mean(r_mom_x**2) + torch.mean(r_mom_y**2)
    
    return loss


def bc_loss(model, X_bc, bc_values):
    x = X_bc[:, 0]
    y = X_bc[:, 1]
    t = X_bc[:, 2]
    f0 = X_bc[:, 3]
    
    u_true = bc_values[:, 0]
    v_true = bc_values[:, 1]
    
    u_pred, v_pred, _ = model(x, y, t, f0)
    
    loss_u = torch.mean((u_pred - u_true)**2)
    loss_v = torch.mean((v_pred - v_true)**2)
    
    return loss_u + loss_v


# ============================================================
# 5. TRAINING (unchanged)
# ============================================================

def train_pinn(model, X_data, Y_data, X_coll, X_bc, bc_values):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    
    X_data_t = torch.tensor(X_data, dtype=torch.float32).to(device)
    Y_data_t = torch.tensor(Y_data, dtype=torch.float32).to(device)
    X_coll_t = torch.tensor(X_coll, dtype=torch.float32).to(device)
    X_bc_t = torch.tensor(X_bc, dtype=torch.float32).to(device)
    bc_values_t = torch.tensor(bc_values, dtype=torch.float32).to(device)
    
    history = {'total': [], 'data': [], 'pde': [], 'bc': []}
    
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Training PINN v2 (More Data, Data-Focused)")
    print("="*60)
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        L_data = data_loss(model, X_data_t, Y_data_t)
        L_pde = pde_loss(model, X_coll_t)
        L_bc = bc_loss(model, X_bc_t, bc_values_t)
        
        L_total = LAMBDA_DATA * L_data + LAMBDA_PDE * L_pde + LAMBDA_BC * L_bc
        
        L_total.backward()
        optimizer.step()
        
        history['total'].append(L_total.item())
        history['data'].append(L_data.item())
        history['pde'].append(L_pde.item())
        history['bc'].append(L_bc.item())
        
        scheduler.step(L_total)
        
        if L_total.item() < best_loss:
            best_loss = L_total.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        if epoch % 500 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d} | Total: {L_total.item():.6e} | Data: {L_data.item():.6e} | "
                  f"PDE: {L_pde.item():.6e} | BC: {L_bc.item():.6e} | Time: {elapsed:.1f}s")
    
    model.load_state_dict(best_state)
    print(f"\nBest total loss: {best_loss:.6e}")
    
    return model, history


# ============================================================
# 6. EVALUATION (unchanged)
# ============================================================

def evaluate_pinn(model, x_cells, y_cells, t_array, traj_true, f0):
    model.eval()
    n_cells = len(x_cells)
    n_times = len(t_array)
    
    u_pred_all = np.zeros((n_times, n_cells))
    v_pred_all = np.zeros((n_times, n_cells))
    
    with torch.no_grad():
        for i, t_val in enumerate(t_array):
            x_t = torch.tensor(x_cells, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_cells, dtype=torch.float32).to(device)
            t_t = torch.full_like(x_t, t_val)
            f0_t = torch.full_like(x_t, f0)
            
            u, v, p = model(x_t, y_t, t_t, f0_t)
            
            u_pred_all[i] = u.cpu().numpy()
            v_pred_all[i] = v.cpu().numpy()
    
    u_true = traj_true[:, :, 0]
    v_true = traj_true[:, :, 1]
    
    U_mag_true = np.sqrt(u_true**2 + v_true**2)
    U_mag_pred = np.sqrt(u_pred_all**2 + v_pred_all**2)
    
    rel_errors = np.linalg.norm(U_mag_pred - U_mag_true, axis=1) / (np.linalg.norm(U_mag_true, axis=1) + 1e-10)
    
    mean_error = np.mean(rel_errors) * 100
    max_error = np.max(rel_errors) * 100
    
    ss_res = np.sum((U_mag_pred - U_mag_true)**2)
    ss_tot = np.sum((U_mag_true - U_mag_true.mean())**2)
    r2 = 1 - ss_res / ss_tot
    
    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'r2': r2,
        'u_pred': u_pred_all,
        'v_pred': v_pred_all,
        'rel_errors': rel_errors
    }


# ============================================================
# 7. VISUALIZATION (unchanged)
# ============================================================

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.semilogy(history['total'], label='Total')
    ax.semilogy(history['data'], label='Data')
    ax.semilogy(history['pde'], label='PDE')
    ax.semilogy(history['bc'], label='BC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1]
    ax.semilogy(history['total'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Convergence')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('pinn_v2_training_loss.png', dpi=150)
    print("Saved: pinn_v2_training_loss.png")
    plt.show()


def plot_pinn_results(x_cells, y_cells, t_array, traj_true, results, f0, save_prefix):
    time_indices = [0, len(t_array)//4, len(t_array)//2, 3*len(t_array)//4, len(t_array)-1]
    
    fig, axes = plt.subplots(3, len(time_indices), figsize=(18, 10))
    
    for col, tidx in enumerate(time_indices):
        t_val = t_array[tidx]
        
        U_mag_true = np.sqrt(traj_true[tidx, :, 0]**2 + traj_true[tidx, :, 1]**2)
        U_mag_pred = np.sqrt(results['u_pred'][tidx]**2 + results['v_pred'][tidx]**2)
        error = np.abs(U_mag_pred - U_mag_true)
        
        vmin, vmax = U_mag_true.min(), U_mag_true.max()
        
        axes[0, col].scatter(x_cells, y_cells, c=U_mag_true, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f't = {t_val:.1f}s')
        axes[0, col].set_aspect('equal')
        if col == 0:
            axes[0, col].set_ylabel('CFD', fontsize=12)
        
        axes[1, col].scatter(x_cells, y_cells, c=U_mag_pred, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
        axes[1, col].set_aspect('equal')
        if col == 0:
            axes[1, col].set_ylabel('PINN', fontsize=12)
        
        axes[2, col].scatter(x_cells, y_cells, c=error, cmap='Reds', s=3)
        axes[2, col].set_aspect('equal')
        if col == 0:
            axes[2, col].set_ylabel('Error', fontsize=12)
    
    plt.suptitle(f'PINN v2 Results (f₀ = {f0} Hz) - Mean Error: {results["mean_error"]:.2f}%', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=150)
    print(f"Saved: {save_prefix}.png")
    plt.show()


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("="*60)
    print("PINN v2 for Oscillating Cylinder Flow")
    print("="*60)
    print(f"\nConfig changes from v1:")
    print(f"  - TIME_DOWNSAMPLE: 5 -> 1 (all snapshots)")
    print(f"  - HIDDEN_DIM: 64 -> 128")
    print(f"  - LAMBDA_DATA: 1.0 -> 10.0")
    print(f"  - LAMBDA_PDE: 0.1 -> 0.01")
    print(f"  - LAMBDA_BC: 1.0 -> 0.1")
    
    print("\n--- Loading Data ---")
    X_data, Y_data, x_cells, y_cells, t_array, traj1, traj2 = load_and_prepare_data()
    
    print("\n--- Generating Collocation Points ---")
    t_min, t_max = t_array.min(), t_array.max()
    X_coll = generate_collocation_points(x_cells, y_cells, t_min, t_max)
    
    print("\n--- Generating BC Points ---")
    X_bc, bc_types, bc_values = generate_bc_points(t_min, t_max)
    
    print("\n--- Creating Model ---")
    model = PINN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    model, history = train_pinn(model, X_data, Y_data, X_coll, X_bc, bc_values)
    
    plot_training_history(history)
    
    data_f01 = np.load(DATA_F01)
    data_f03 = np.load(DATA_F03)
    traj1_unnorm = data_f01['traj'] * data_f01['std'] + data_f01['mean']
    traj2_unnorm = data_f03['traj'] * data_f03['std'] + data_f03['mean']
    
    print("\n--- Evaluation ---")
    
    results_f01 = evaluate_pinn(model, x_cells, y_cells, t_array, traj1_unnorm, F0_1)
    print(f"\n=== Results (f0=0.1) ===")
    print(f"Mean relative error: {results_f01['mean_error']:.2f}%")
    print(f"Max relative error:  {results_f01['max_error']:.2f}%")
    print(f"R² score:            {results_f01['r2']:.4f}")
    
    results_f03 = evaluate_pinn(model, x_cells, y_cells, t_array, traj2_unnorm, F0_2)
    print(f"\n=== Results (f0=0.3) ===")
    print(f"Mean relative error: {results_f03['mean_error']:.2f}%")
    print(f"Max relative error:  {results_f03['max_error']:.2f}%")
    print(f"R² score:            {results_f03['r2']:.4f}")
    
    plot_pinn_results(x_cells, y_cells, t_array, traj1_unnorm, results_f01, F0_1, 'pinn_v2_results_f01')
    plot_pinn_results(x_cells, y_cells, t_array, traj2_unnorm, results_f03, F0_2, 'pinn_v2_results_f03')
    
    torch.save({
        'model_state': model.state_dict(),
        'history': history,
    }, 'pinn_v2_model.pt')
    print("\nSaved: pinn_v2_model.pt")
    
    print("\n" + "="*60)
    print("PINN v2 Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()