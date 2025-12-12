"""
PINN v2 Test on Unseen Frequency f0 = 0.2 Hz
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import re

# ============================================================
# CONFIG
# ============================================================
DATA_F02 = "data/dataset_f02.npz"
CELL_CENTERS_FILE = "OpenFOAM/Re_100_f0_01/2/C"
MODEL_PATH = "pinn_v2_model.pt"

HIDDEN_LAYERS = 6
HIDDEN_DIM = 128
F0_TEST = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# PARSE CELL CENTERS
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

# ============================================================
# PINN MODEL (must match training)
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
    
    def forward(self, x, y, t, f0):
        inputs = torch.stack([x, y, t, f0], dim=1)
        outputs = self.net(inputs)
        u = outputs[:, 0]
        v = outputs[:, 1]
        p = outputs[:, 2]
        return u, v, p

# ============================================================
# LOAD MODEL
# ============================================================
print("\n--- Loading Model ---")
model = PINN().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model.eval()
print("Model loaded successfully!")

# ============================================================
# LOAD TEST DATA
# ============================================================
print("\n--- Loading Test Data ---")
cell_centers = parse_openfoam_cell_centers(CELL_CENTERS_FILE)
x_cells = cell_centers[:, 0]
y_cells = cell_centers[:, 1]
n_cells = len(x_cells)

data_f02 = np.load(DATA_F02)
t_array = data_f02['t']

# f0=0.2 data has different format
if 'U_flat' in data_f02.files:
    # U_flat shape: (T, N*2) -> reshape to (T, N, 2)
    U_flat = data_f02['U_flat']
    traj_f02 = U_flat.reshape(len(t_array), n_cells, 2)
elif 'std' in data_f02.files:
    traj_f02 = data_f02['traj'] * data_f02['std'] + data_f02['mean']
else:
    traj_f02 = data_f02['traj']

n_times = len(t_array)
print(f"Test data: {n_times} timesteps, {n_cells} cells")
print(f"Test frequency: f0 = {F0_TEST} Hz (UNSEEN)")

# ============================================================
# EVALUATE
# ============================================================
print("\n--- Running Predictions ---")
u_pred_all = np.zeros((n_times, n_cells))
v_pred_all = np.zeros((n_times, n_cells))

with torch.no_grad():
    for i, t_val in enumerate(t_array):
        x_t = torch.tensor(x_cells, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_cells, dtype=torch.float32).to(device)
        t_t = torch.full_like(x_t, t_val)
        f0_t = torch.full_like(x_t, F0_TEST)
        
        u, v, p = model(x_t, y_t, t_t, f0_t)
        
        u_pred_all[i] = u.cpu().numpy()
        v_pred_all[i] = v.cpu().numpy()
        
        if (i + 1) % 30 == 0:
            print(f"  Processed {i+1}/{n_times} timesteps")

# ============================================================
# COMPUTE METRICS
# ============================================================
print("\n--- Computing Metrics ---")
u_true = traj_f02[:, :, 0]
v_true = traj_f02[:, :, 1]

U_mag_true = np.sqrt(u_true**2 + v_true**2)
U_mag_pred = np.sqrt(u_pred_all**2 + v_pred_all**2)

# Per-timestep relative errors
rel_errors = np.linalg.norm(U_mag_pred - U_mag_true, axis=1) / (np.linalg.norm(U_mag_true, axis=1) + 1e-10)

mean_error = np.mean(rel_errors) * 100
max_error = np.max(rel_errors) * 100

# R² score
ss_res = np.sum((U_mag_pred - U_mag_true)**2)
ss_tot = np.sum((U_mag_true - U_mag_true.mean())**2)
r2 = 1 - ss_res / ss_tot

print("\n" + "="*60)
print(f"PINN v2 Results on UNSEEN f0 = {F0_TEST} Hz")
print("="*60)
print(f"Mean Relative Error: {mean_error:.2f}%")
print(f"Max Relative Error:  {max_error:.2f}%")
print(f"R² Score:            {r2:.4f}")
print("="*60)

# ============================================================
# VISUALIZATION
# ============================================================
print("\n--- Generating Visualization ---")
time_indices = [0, n_times//4, n_times//2, 3*n_times//4, n_times-1]

fig, axes = plt.subplots(3, len(time_indices), figsize=(18, 10))

for col, tidx in enumerate(time_indices):
    t_val = t_array[tidx]
    
    U_true = U_mag_true[tidx]
    U_pred = U_mag_pred[tidx]
    error = np.abs(U_pred - U_true)
    
    vmin, vmax = U_true.min(), U_true.max()
    
    axes[0, col].scatter(x_cells, y_cells, c=U_true, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f't = {t_val:.1f}s')
    axes[0, col].set_aspect('equal')
    if col == 0:
        axes[0, col].set_ylabel('CFD', fontsize=12)
    
    axes[1, col].scatter(x_cells, y_cells, c=U_pred, cmap='coolwarm', s=3, vmin=vmin, vmax=vmax)
    axes[1, col].set_aspect('equal')
    if col == 0:
        axes[1, col].set_ylabel('PINN', fontsize=12)
    
    axes[2, col].scatter(x_cells, y_cells, c=error, cmap='Reds', s=3)
    axes[2, col].set_aspect('equal')
    if col == 0:
        axes[2, col].set_ylabel('Error', fontsize=12)

plt.suptitle(f'PINN v2 Results (f₀ = {F0_TEST} Hz UNSEEN) - Mean Error: {mean_error:.2f}%; R² = {r2:.4f}', fontsize=14)
plt.tight_layout()
plt.savefig('pinn_v2_results_f02_unseen.png', dpi=150)
print("\nSaved: pinn_v2_results_f02_unseen.png")
plt.show()

print("\nDone!")