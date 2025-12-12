"""
Neural ODE with POD - TRAJECTORY TRAINING
==========================================
Key fix: Train on trajectory segments, not single steps.
This teaches the network to be stable under integration.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

DATA_F01 = "dataset_f01.npz"
DATA_F03 = "dataset_f03.npz"

N_MODES = 15
EPOCHS = 800
BATCH_SIZE = 16
LR = 1e-3
HIDDEN_DIM = 128
SEQ_LEN = 20          # Train on sequences of 20 steps
WEIGHT_DECAY = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. DATA LOADING AND POD
# ============================================================

def load_data():
    d1 = np.load(DATA_F01)
    d2 = np.load(DATA_F03)
    
    t1, traj1 = d1['t'], d1['traj'][:, :, :2]
    t2, traj2 = d2['t'], d2['traj'][:, :, :2]
    
    dt = float(t1[1] - t1[0])
    
    print(f"Loaded trajectories: {traj1.shape}, {traj2.shape}")
    print(f"Time: {t1[0]} to {t1[-1]}, dt = {dt}")
    
    return t1, traj1, t2, traj2, dt


def compute_pod(traj1, traj2, n_modes):
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
    
    energy = S**2 / np.sum(S**2)
    cumulative = np.cumsum(energy)
    print(f"\nPOD with {n_modes} modes captures {cumulative[n_modes-1]*100:.2f}% of energy")
    
    recon1 = U_mean + a1 @ Phi.T
    recon_error = np.linalg.norm(traj1_flat - recon1) / np.linalg.norm(traj1_flat)
    print(f"POD reconstruction error: {recon_error*100:.2f}%")
    
    return U_mean, Phi, a1, a2, S[:n_modes]


def reconstruct_velocity(a, U_mean, Phi, original_shape):
    if a.ndim == 1:
        a = a.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    U_flat = U_mean + a @ Phi.T
    U = U_flat.reshape(-1, *original_shape)
    
    if squeeze:
        U = U[0]
    
    return U


# ============================================================
# 2. NEURAL ODE MODEL
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
        
        # Small initialization for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, a, f0):
        """
        Args:
            a: (batch, n_modes)
            f0: (batch, 1)
        Returns:
            da/dt: (batch, n_modes)
        """
        inp = torch.cat([a, f0], dim=-1)
        return self.net(inp)


# ============================================================
# 3. TRAJECTORY DATASET
# ============================================================

class TrajectoryDataset(Dataset):
    """
    Dataset of trajectory segments for training.
    Each sample is a sequence of SEQ_LEN consecutive POD coefficients.
    """
    def __init__(self, a1, a2, f0_1, f0_2, seq_len):
        self.sequences = []
        self.f0_values = []
        
        # Create overlapping sequences from trajectory 1
        for i in range(len(a1) - seq_len):
            self.sequences.append(a1[i:i+seq_len+1])  # +1 for target
            self.f0_values.append(f0_1)
        
        # Create overlapping sequences from trajectory 2
        for i in range(len(a2) - seq_len):
            self.sequences.append(a2[i:i+seq_len+1])
            self.f0_values.append(f0_2)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.f0_values = np.array(self.f0_values, dtype=np.float32)
        
        print(f"Dataset: {len(self.sequences)} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])
        f0 = torch.tensor([self.f0_values[idx]])
        return seq, f0


# ============================================================
# 4. INTEGRATION (DIFFERENTIABLE)
# ============================================================

def integrate_rk4(ode_func, a0, f0, n_steps, dt):
    """
    Integrate ODE using RK4 - differentiable for training.
    
    Args:
        ode_func: neural network
        a0: initial condition (batch, n_modes)
        f0: frequency parameter (batch, 1)
        n_steps: number of integration steps
        dt: time step
    
    Returns:
        trajectory: (batch, n_steps+1, n_modes)
    """
    batch_size = a0.shape[0]
    trajectory = [a0]
    
    a = a0
    for _ in range(n_steps):
        k1 = ode_func(a, f0)
        k2 = ode_func(a + 0.5 * dt * k1, f0)
        k3 = ode_func(a + 0.5 * dt * k2, f0)
        k4 = ode_func(a + dt * k3, f0)
        a = a + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(a)
    
    return torch.stack(trajectory, dim=1)  # (batch, n_steps+1, n_modes)


# ============================================================
# 5. TRAINING WITH TRAJECTORY LOSS
# ============================================================

def train_model(ode_func, train_loader, val_loader, dt, epochs=EPOCHS):
    ode_func.to(device)
    optimizer = torch.optim.AdamW(ode_func.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    
    best_val_loss = float('inf')
    best_state = None
    train_losses = []
    val_losses = []
    
    # Early stopping
    patience = 100
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Training
        ode_func.train()
        train_loss = 0.0
        
        for seq_batch, f0_batch in train_loader:
            seq_batch = seq_batch.to(device)
            f0_batch = f0_batch.to(device)
            
            a0 = seq_batch[:, 0, :]
            target = seq_batch[:, 1:, :]
            
            pred_traj = integrate_rk4(ode_func, a0, f0_batch, SEQ_LEN, dt)
            pred = pred_traj[:, 1:, :]
            
            loss = torch.mean((pred - target)**2)
            
            if pred.shape[1] > 1:
                pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
                target_diff = target[:, 1:, :] - target[:, :-1, :]
                loss = loss + 0.1 * torch.mean((pred_diff - target_diff)**2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * seq_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        ode_func.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_batch, f0_batch in val_loader:
                seq_batch = seq_batch.to(device)
                f0_batch = f0_batch.to(device)
                
                a0 = seq_batch[:, 0, :]
                target = seq_batch[:, 1:, :]
                
                pred_traj = integrate_rk4(ode_func, a0, f0_batch, SEQ_LEN, dt)
                pred = pred_traj[:, 1:, :]
                
                loss = torch.mean((pred - target)**2)
                val_loss += loss.item() * seq_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in ode_func.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.6e} | Val: {val_loss:.6e}")
    
    ode_func.load_state_dict(best_state)
    print(f"Best validation loss: {best_val_loss:.6e}")
    
    return ode_func, train_losses, val_losses

# ============================================================
# 6. ROLLOUT
# ============================================================

def rollout(ode_func, a0, t_span, dt, f0):
    ode_func.eval()
    
    a = torch.from_numpy(a0.astype(np.float32)).unsqueeze(0).to(device)
    f0_tensor = torch.tensor([[f0]], dtype=torch.float32).to(device)
    trajectory = [a0]
    
    with torch.no_grad():
        for i in range(len(t_span) - 1):
            k1 = ode_func(a, f0_tensor)
            k2 = ode_func(a + 0.5 * dt * k1, f0_tensor)
            k3 = ode_func(a + 0.5 * dt * k2, f0_tensor)
            k4 = ode_func(a + dt * k3, f0_tensor)
            a = a + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(a.cpu().numpy()[0])
    
    return np.array(trajectory)


# ============================================================
# 7. METRICS
# ============================================================

def compute_errors(true_traj, pred_traj):
    diff = true_traj - pred_traj
    rel_errors = np.linalg.norm(diff.reshape(len(diff), -1), axis=1) / \
                 (np.linalg.norm(true_traj.reshape(len(true_traj), -1), axis=1) + 1e-10)
    
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((true_traj - true_traj.mean())**2)
    r2 = 1 - ss_res / ss_tot
    
    return {
        'rel_errors': rel_errors,
        'mean_rel_error': np.mean(rel_errors),
        'max_rel_error': np.max(rel_errors),
        'r2': r2
    }


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Neural ODE with POD - TRAJECTORY TRAINING")
    print("=" * 60)
    
    F0_1, F0_2 = 0.1, 0.3
    
    t1, traj1, t2, traj2, dt = load_data()
    original_shape = traj1.shape[1:]
    
    print("\n--- Computing POD ---")
    U_mean, Phi, a1, a2, _ = compute_pod(traj1, traj2, N_MODES)
    
    # Normalize
    a_all = np.vstack([a1, a2])
    a_mean = a_all.mean(axis=0)
    a_std = a_all.std(axis=0) + 1e-8
    
    a1_norm = (a1 - a_mean) / a_std
    a2_norm = (a2 - a_mean) / a_std
    
    f0_mean = (F0_1 + F0_2) / 2
    f0_std = (F0_2 - F0_1) / 2 + 1e-8
    f0_1_norm = (F0_1 - f0_mean) / f0_std
    f0_2_norm = (F0_2 - f0_mean) / f0_std
    
    # Normalize dt for numerical stability
    dt_norm = dt / a_std.mean()  # Scale dt with data
    dt_norm = dt  # Actually keep original dt
    
    print(f"\nf0 normalized: {F0_1} -> {f0_1_norm:.2f}, {F0_2} -> {f0_2_norm:.2f}")
    
    # Create dataset
    print("\n--- Creating Dataset ---")
    dataset = TrajectoryDataset(a1_norm, a2_norm, f0_1_norm, f0_2_norm, SEQ_LEN)
    
    # Train/val split
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    
    train_dataset = torch.utils.data.Subset(dataset, indices[:split])
    val_dataset = torch.utils.data.Subset(dataset, indices[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Train
    print("\n--- Training Neural ODE ---")
    ode_func = ODEFunc(N_MODES, HIDDEN_DIM)
    print(f"Model parameters: {sum(p.numel() for p in ode_func.parameters()):,}")
    
    ode_func, train_losses, val_losses = train_model(
        ode_func, train_loader, val_loader, dt, epochs=EPOCHS
    )
    
    # Evaluate
    print("\n--- Rollout and Evaluation ---")
    
    # f0 = 0.1
    a_pred_norm = rollout(ode_func, a1_norm[0], t1, dt, f0_1_norm)
    a_pred = a_pred_norm * a_std + a_mean
    U_pred = reconstruct_velocity(a_pred, U_mean, Phi, original_shape)
    errors = compute_errors(traj1, U_pred)
    
    print(f"\n=== Results (f0=0.1) ===")
    print(f"Mean relative error: {errors['mean_rel_error']*100:.2f}%")
    print(f"Max relative error:  {errors['max_rel_error']*100:.2f}%")
    print(f"R² score:            {errors['r2']:.4f}")
    
    # f0 = 0.3
    a_pred_norm_2 = rollout(ode_func, a2_norm[0], t2, dt, f0_2_norm)
    a_pred_2 = a_pred_norm_2 * a_std + a_mean
    U_pred_2 = reconstruct_velocity(a_pred_2, U_mean, Phi, original_shape)
    errors_2 = compute_errors(traj2, U_pred_2)
    
    print(f"\n=== Results (f0=0.3) ===")
    print(f"Mean relative error: {errors_2['mean_rel_error']*100:.2f}%")
    print(f"Max relative error:  {errors_2['max_rel_error']*100:.2f}%")
    print(f"R² score:            {errors_2['r2']:.4f}")
    
    # ============================================================
    # PLOTTING
    # ============================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    ax.semilogy(train_losses, label='Train')
    ax.semilogy(val_losses, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)
    
    ax = axes[0, 1]
    for i in range(3):
        ax.plot(t1, a1[:, i], '-', label=f'True {i+1}', alpha=0.8)
        ax.plot(t1, a_pred[:, i], '--', label=f'Pred {i+1}', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('POD coefficient')
    ax.set_title('POD Coefficients (f0=0.1)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)
    
    ax = axes[0, 2]
    ax.plot(t1, errors['rel_errors'] * 100)
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title(f'Error (f0=0.1), Mean={errors["mean_rel_error"]*100:.2f}%')
    ax.grid(True)
    
    cell_idx = 1000
    ax = axes[1, 0]
    ax.plot(t1, traj1[:, cell_idx, 0], 'b-', label='CFD', linewidth=2)
    ax.plot(t1, U_pred[:, cell_idx, 0], 'r--', label='Neural ODE', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('u_x')
    ax.set_title(f'Velocity at Cell {cell_idx} (f0=0.1)')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(t2, a2[:, i], '-', label=f'True {i+1}', alpha=0.8)
        ax.plot(t2, a_pred_2[:, i], '--', label=f'Pred {i+1}', alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('POD coefficient')
    ax.set_title('POD Coefficients (f0=0.3)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)
    
    ax = axes[1, 2]
    ax.plot(t2, traj2[:, cell_idx, 0], 'b-', label='CFD', linewidth=2)
    ax.plot(t2, U_pred_2[:, cell_idx, 0], 'r--', label='Neural ODE', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('u_x')
    ax.set_title(f'Velocity at Cell {cell_idx} (f0=0.3)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('neural_ode_results_v3.png', dpi=150)
    print("\nSaved: neural_ode_results_v3.png")
    plt.show()
    
    torch.save({
        'model_state': ode_func.state_dict(),
        'U_mean': U_mean, 'Phi': Phi,
        'a_mean': a_mean, 'a_std': a_std,
        'f0_mean': f0_mean, 'f0_std': f0_std,
        'n_modes': N_MODES, 'original_shape': original_shape,
    }, 'neural_ode_model_v3.pt')
    print("Saved: neural_ode_model_v3.pt")


if __name__ == "__main__":
    main()