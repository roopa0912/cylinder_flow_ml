"""
VAE + Neural ODE for Oscillating Cylinder Flow
===============================================
Nonlinear dimensionality reduction with learned dynamics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import re

# ============================================================
# CONFIG
# ============================================================

DATA_F01 = "dataset_f01.npz"
DATA_F03 = "dataset_f03.npz"
CELL_CENTERS_FILE = "Re_100_f0_01/2/C"

# VAE Architecture
INPUT_DIM = 6912          # 3456 cells × 2 components (u, v)
LATENT_DIM = 16           # Similar to POD modes
HIDDEN_DIM_VAE = 256      # VAE hidden layer size

# Neural ODE Architecture  
HIDDEN_DIM_ODE = 128      # ODE network hidden size

# Training - Phase 1 (VAE)
VAE_EPOCHS = 500
VAE_LR = 1e-3
VAE_BATCH_SIZE = 32
KL_WEIGHT = 0.001         # β in β-VAE (small = focus on reconstruction)

# Training - Phase 2 (Neural ODE)
ODE_EPOCHS = 500
ODE_LR = 1e-3
ODE_BATCH_SIZE = 16
SEQ_LEN = 20

# Frequencies
F0_1, F0_2 = 0.1, 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data():
    """Load and prepare velocity data."""
    d1 = np.load(DATA_F01)
    d2 = np.load(DATA_F03)
    
    t1, traj1 = d1['t'], d1['traj'][:, :, :2]  # (151, 3456, 2)
    t2, traj2 = d2['t'], d2['traj'][:, :, :2]
    
    # Unnormalize
    traj1 = traj1 * d1['std'] + d1['mean']
    traj2 = traj2 * d2['std'] + d2['mean']
    
    # Flatten spatial dimensions
    traj1_flat = traj1.reshape(traj1.shape[0], -1)  # (151, 6912)
    traj2_flat = traj2.reshape(traj2.shape[0], -1)
    
    dt = float(t1[1] - t1[0])
    
    print(f"Loaded data: {traj1_flat.shape}, {traj2_flat.shape}")
    print(f"Time: {t1[0]} to {t1[-1]}, dt = {dt}")
    print(f"Velocity range: [{traj1_flat.min():.3f}, {traj1_flat.max():.3f}]")
    
    return t1, traj1_flat, traj2_flat, traj1.shape[1:], dt


def normalize_data(traj1, traj2):
    """Normalize data to zero mean, unit variance."""
    all_data = np.vstack([traj1, traj2])
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0) + 1e-8
    
    traj1_norm = (traj1 - mean) / std
    traj2_norm = (traj2 - mean) / std
    
    return traj1_norm, traj2_norm, mean, std


# ============================================================
# 2. VAE MODEL
# ============================================================

class FlowVAE(nn.Module):
    """Variational Autoencoder for flow field compression."""
    
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM_VAE):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: U → hidden → (μ, log_σ²)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder: z → hidden → U
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def get_latent(self, x):
        """Get latent representation (mean, for inference)."""
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon, x, mu, logvar, kl_weight=KL_WEIGHT):
    """VAE loss = Reconstruction + β * KL divergence."""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


# ============================================================
# 3. NEURAL ODE (operates on latent space)
# ============================================================

class LatentODEFunc(nn.Module):
    """Neural ODE function for latent dynamics: dz/dt = f(z, f₀)."""
    
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM_ODE):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for f₀
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Initialize with small weights for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, z, f0):
        """Predict dz/dt."""
        inp = torch.cat([z, f0], dim=-1)
        return self.net(inp)


class LatentTrajectoryDataset(Dataset):
    """Dataset of latent trajectory segments."""
    
    def __init__(self, z1, z2, f0_1, f0_2, seq_len):
        self.sequences = []
        self.f0_values = []
        
        for i in range(len(z1) - seq_len):
            self.sequences.append(z1[i:i+seq_len+1])
            self.f0_values.append(f0_1)
        
        for i in range(len(z2) - seq_len):
            self.sequences.append(z2[i:i+seq_len+1])
            self.f0_values.append(f0_2)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.f0_values = np.array(self.f0_values, dtype=np.float32)
        
        print(f"Latent dataset: {len(self.sequences)} sequences of length {seq_len}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor([self.f0_values[idx]])


def integrate_ode(ode_func, z0, f0, n_steps, dt):
    """RK4 integration in latent space."""
    trajectory = [z0]
    z = z0
    
    for _ in range(n_steps):
        k1 = ode_func(z, f0)
        k2 = ode_func(z + 0.5 * dt * k1, f0)
        k3 = ode_func(z + 0.5 * dt * k2, f0)
        k4 = ode_func(z + dt * k3, f0)
        z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(z)
    
    return torch.stack(trajectory, dim=1)


# ============================================================
# 4. TRAINING PHASE 1: VAE
# ============================================================

def train_vae(vae, traj1_norm, traj2_norm):
    """Train VAE for reconstruction."""
    print("\n" + "="*60)
    print("Phase 1: Training VAE")
    print("="*60)
    
    vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    
    # Combine data
    all_data = np.vstack([traj1_norm, traj2_norm])
    dataset = TensorDataset(torch.tensor(all_data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=VAE_BATCH_SIZE, shuffle=True)
    
    history = {'total': [], 'recon': [], 'kl': []}
    best_loss = float('inf')
    best_state = None
    
    for epoch in range(1, VAE_EPOCHS + 1):
        vae.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for batch in loader:
            x = batch[0].to(device)
            
            recon, mu, logvar, z = vae(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            epoch_recon += recon_loss.item() * x.size(0)
            epoch_kl += kl_loss.item() * x.size(0)
        
        epoch_loss /= len(all_data)
        epoch_recon /= len(all_data)
        epoch_kl /= len(all_data)
        
        history['total'].append(epoch_loss)
        history['recon'].append(epoch_recon)
        history['kl'].append(epoch_kl)
        
        scheduler.step(epoch_loss)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in vae.state_dict().items()}
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.6e} | Recon: {epoch_recon:.6e} | KL: {epoch_kl:.6e}")
    
    vae.load_state_dict(best_state)
    print(f"Best VAE loss: {best_loss:.6e}")
    
    return vae, history


# ============================================================
# 5. TRAINING PHASE 2: Neural ODE
# ============================================================

def train_ode(ode_func, vae, traj1_norm, traj2_norm, dt):
    """Train Neural ODE on latent trajectories."""
    print("\n" + "="*60)
    print("Phase 2: Training Neural ODE on Latent Space")
    print("="*60)
    
    # Encode all data to latent space
    vae.eval()
    with torch.no_grad():
        z1 = vae.get_latent(torch.tensor(traj1_norm, dtype=torch.float32).to(device)).cpu().numpy()
        z2 = vae.get_latent(torch.tensor(traj2_norm, dtype=torch.float32).to(device)).cpu().numpy()
    
    print(f"Latent trajectories: {z1.shape}, {z2.shape}")
    print(f"Latent range: [{z1.min():.3f}, {z1.max():.3f}]")
    
    # Normalize frequency
    f0_mean = (F0_1 + F0_2) / 2
    f0_std = (F0_2 - F0_1) / 2 + 1e-8
    f0_1_norm = (F0_1 - f0_mean) / f0_std
    f0_2_norm = (F0_2 - f0_mean) / f0_std
    
    # Create dataset
    dataset = LatentTrajectoryDataset(z1, z2, f0_1_norm, f0_2_norm, SEQ_LEN)
    
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    split = int(0.8 * n_samples)
    
    train_dataset = torch.utils.data.Subset(dataset, indices[:split])
    val_dataset = torch.utils.data.Subset(dataset, indices[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=ODE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ODE_BATCH_SIZE)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Training
    ode_func.to(device)
    optimizer = torch.optim.AdamW(ode_func.parameters(), lr=ODE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    
    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    best_state = None
    patience = 100
    patience_counter = 0
    
    for epoch in range(1, ODE_EPOCHS + 1):
        # Train
        ode_func.train()
        train_loss = 0
        
        for seq_batch, f0_batch in train_loader:
            seq_batch = seq_batch.to(device)
            f0_batch = f0_batch.to(device)
            
            z0 = seq_batch[:, 0, :]
            target = seq_batch[:, 1:, :]
            
            pred_traj = integrate_ode(ode_func, z0, f0_batch, SEQ_LEN, dt)
            pred = pred_traj[:, 1:, :]
            
            loss = F.mse_loss(pred, target)
            
            # Smoothness regularization
            if pred.shape[1] > 1:
                pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
                target_diff = target[:, 1:, :] - target[:, :-1, :]
                loss = loss + 0.1 * F.mse_loss(pred_diff, target_diff)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * seq_batch.size(0)
        
        train_loss /= len(train_dataset)
        history['train'].append(train_loss)
        
        # Validate
        ode_func.eval()
        val_loss = 0
        with torch.no_grad():
            for seq_batch, f0_batch in val_loader:
                seq_batch = seq_batch.to(device)
                f0_batch = f0_batch.to(device)
                
                z0 = seq_batch[:, 0, :]
                target = seq_batch[:, 1:, :]
                
                pred_traj = integrate_ode(ode_func, z0, f0_batch, SEQ_LEN, dt)
                pred = pred_traj[:, 1:, :]
                
                loss = F.mse_loss(pred, target)
                val_loss += loss.item() * seq_batch.size(0)
        
        val_loss /= len(val_dataset)
        history['val'].append(val_loss)
        
        scheduler.step()
        
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
    print(f"Best val loss: {best_val_loss:.6e}")
    
    return ode_func, history, z1, z2, f0_mean, f0_std


# ============================================================
# 6. EVALUATION
# ============================================================

def rollout_vae_ode(vae, ode_func, z0, t_span, dt, f0_norm):
    """Rollout in latent space then decode."""
    vae.eval()
    ode_func.eval()
    
    z = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).to(device)
    f0_tensor = torch.tensor([[f0_norm]], dtype=torch.float32).to(device)
    
    z_traj = [z0]
    
    with torch.no_grad():
        for _ in range(len(t_span) - 1):
            k1 = ode_func(z, f0_tensor)
            k2 = ode_func(z + 0.5 * dt * k1, f0_tensor)
            k3 = ode_func(z + 0.5 * dt * k2, f0_tensor)
            k4 = ode_func(z + dt * k3, f0_tensor)
            z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            z_traj.append(z.cpu().numpy()[0])
    
    z_traj = np.array(z_traj)
    
    # Decode
    with torch.no_grad():
        z_tensor = torch.tensor(z_traj, dtype=torch.float32).to(device)
        U_pred_norm = vae.decode(z_tensor).cpu().numpy()
    
    return z_traj, U_pred_norm


def compute_errors(true_traj, pred_traj):
    """Compute error metrics."""
    diff = true_traj - pred_traj
    rel_errors = np.linalg.norm(diff, axis=1) / (np.linalg.norm(true_traj, axis=1) + 1e-10)
    
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
# 7. MAIN
# ============================================================

def main():
    print("="*60)
    print("VAE + Neural ODE for Oscillating Cylinder Flow")
    print("="*60)
    
    # Load data
    print("\n--- Loading Data ---")
    t_array, traj1_flat, traj2_flat, original_shape, dt = load_data()
    
    # Normalize
    traj1_norm, traj2_norm, data_mean, data_std = normalize_data(traj1_flat, traj2_flat)
    print(f"Normalized data range: [{traj1_norm.min():.3f}, {traj1_norm.max():.3f}]")
    
    # Phase 1: Train VAE
    vae = FlowVAE()
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    vae, vae_history = train_vae(vae, traj1_norm, traj2_norm)
    
    # Phase 2: Train Neural ODE
    ode_func = LatentODEFunc()
    print(f"ODE parameters: {sum(p.numel() for p in ode_func.parameters()):,}")
    
    ode_func, ode_history, z1, z2, f0_mean, f0_std = train_ode(
        ode_func, vae, traj1_norm, traj2_norm, dt
    )
    
    # Normalize frequencies
    f0_1_norm = (F0_1 - f0_mean) / f0_std
    f0_2_norm = (F0_2 - f0_mean) / f0_std
    
    # Evaluate on training frequencies
    print("\n--- Evaluation ---")
    
    # f0 = 0.1
    z_pred_1, U_pred_norm_1 = rollout_vae_ode(vae, ode_func, z1[0], t_array, dt, f0_1_norm)
    U_pred_1 = U_pred_norm_1 * data_std + data_mean
    U_true_1 = traj1_norm * data_std + data_mean
    errors_1 = compute_errors(U_true_1, U_pred_1)
    
    print(f"\n=== Results (f0=0.1) ===")
    print(f"Mean relative error: {errors_1['mean_rel_error']*100:.2f}%")
    print(f"Max relative error:  {errors_1['max_rel_error']*100:.2f}%")
    print(f"R² score:            {errors_1['r2']:.4f}")
    
    # f0 = 0.3
    z_pred_2, U_pred_norm_2 = rollout_vae_ode(vae, ode_func, z2[0], t_array, dt, f0_2_norm)
    U_pred_2 = U_pred_norm_2 * data_std + data_mean
    U_true_2 = traj2_norm * data_std + data_mean
    errors_2 = compute_errors(U_true_2, U_pred_2)
    
    print(f"\n=== Results (f0=0.3) ===")
    print(f"Mean relative error: {errors_2['mean_rel_error']*100:.2f}%")
    print(f"Max relative error:  {errors_2['max_rel_error']*100:.2f}%")
    print(f"R² score:            {errors_2['r2']:.4f}")
    
    # ============================================================
    # PLOTTING
    # ============================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # VAE training loss
    ax = axes[0, 0]
    ax.semilogy(vae_history['total'], label='Total')
    ax.semilogy(vae_history['recon'], label='Recon')
    ax.semilogy(vae_history['kl'], label='KL')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('VAE Training Loss')
    ax.legend()
    ax.grid(True)
    
    # ODE training loss
    ax = axes[0, 1]
    ax.semilogy(ode_history['train'], label='Train')
    ax.semilogy(ode_history['val'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Neural ODE Training Loss')
    ax.legend()
    ax.grid(True)
    
    # Latent trajectories
    ax = axes[0, 2]
    for i in range(min(4, LATENT_DIM)):
        ax.plot(t_array, z1[:, i], '-', alpha=0.7, label=f'True z{i+1}')
        ax.plot(t_array, z_pred_1[:, i], '--', alpha=0.7, label=f'Pred z{i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Latent value')
    ax.set_title('Latent Trajectories (f0=0.1)')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True)
    
    # Error over time (f0=0.1)
    ax = axes[1, 0]
    ax.plot(t_array, errors_1['rel_errors'] * 100)
    ax.axhline(errors_1['mean_rel_error']*100, color='r', linestyle='--', 
               label=f'Mean: {errors_1["mean_rel_error"]*100:.2f}%')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Error Over Time (f0=0.1)')
    ax.legend()
    ax.grid(True)
    
    # Error over time (f0=0.3)
    ax = axes[1, 1]
    ax.plot(t_array, errors_2['rel_errors'] * 100)
    ax.axhline(errors_2['mean_rel_error']*100, color='r', linestyle='--',
               label=f'Mean: {errors_2["mean_rel_error"]*100:.2f}%')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Error Over Time (f0=0.3)')
    ax.legend()
    ax.grid(True)
    
    # Velocity comparison at a cell
    ax = axes[1, 2]
    cell_idx = 1000
    ax.plot(t_array, U_true_1[:, cell_idx], 'b-', label='CFD', linewidth=2)
    ax.plot(t_array, U_pred_1[:, cell_idx], 'r--', label='VAE+ODE', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.set_title(f'Velocity at Cell {cell_idx} (f0=0.1)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('vae_neural_ode_results.png', dpi=150)
    print("\nSaved: vae_neural_ode_results.png")
    plt.show()
    
    # Save models
    torch.save({
        'vae_state': vae.state_dict(),
        'ode_state': ode_func.state_dict(),
        'data_mean': data_mean,
        'data_std': data_std,
        'f0_mean': f0_mean,
        'f0_std': f0_std,
        'latent_dim': LATENT_DIM,
        'original_shape': original_shape,
    }, 'vae_neural_ode_model.pt')
    print("Saved: vae_neural_ode_model.pt")
    
    print("\n" + "="*60)
    print("VAE + Neural ODE Training Complete!")
    print("="*60)
    
    return vae, ode_func, errors_1, errors_2


if __name__ == "__main__":
    vae, ode_func, errors_1, errors_2 = main()