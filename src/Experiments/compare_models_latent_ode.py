import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from src.dataset import SyntheticDataGenerator
from src.models.neural_ode import *
from src.models.neural_ode.model import LatentODE

# Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mode = '3d'
duration_sec = 250
sampling_rate_hz = 2
num_samples = int(duration_sec * sampling_rate_hz)
time_step = 1 / sampling_rate_hz
noise_variance = 0.1
random_seed = 42

# --- Hyperparameters for the Latent ODE Model ---
latent_dim = 3  # Should match the true latent dimensionality of Lorenz
hidden_dim = 64  # Hidden dimension for the RNN encoder
obs_dim = 10  # Observation dimension
num_epochs = 7000
learning_rate = 1e-3
kl_weight = 0.1  # Weight for the KL divergence term in the loss

# --- Generate Data ---
print(f"Using device: {device}")
gen = SyntheticDataGenerator(
    num_steps=num_samples,
    time_step=time_step,
    noise_variance=noise_variance,
    random_seed=random_seed,
    device=device
)

synthetic_data = gen.generate_lorenz_3d()
t_numpy = synthetic_data["t"]
t = torch.from_numpy(t_numpy).float().to(device)
observation_z = synthetic_data["observation_z"].t()  # Shape: (500, 10)
true_latent = synthetic_data["latent"]  # Shape: (500, 3)
true_latent_torch = true_latent.to(device).t().unsqueeze(1)

# Reshape data for RNN: (seq_len, batch_size, feature_dim)
# We treat the entire time series as a single batch item.
observations_batch = observation_z.unsqueeze(1)  # Shape: (500, 1, 10)

# Plot the ground truth latent dynamics
print("Displaying ground truth Lorenz attractor...")
gen.plot_lorenz_3d(true_latent)
plt.show()

# --- Training Loop ---
model = LatentODE(obs_dim, latent_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
reconstruction_loss_fn = nn.MSELoss()

history = {
    "total_loss": [],
    "kl": [],
    "rec_loss": []
}
print("\nStarting Latent ODE training...")
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()

    reconstructed_obs, z0_mean, z0_logvar, _ = model(observations_batch, t)

    # 1. Reconstruction Loss
    recon_loss = F.mse_loss(reconstructed_obs, observations_batch)

    # 2. KL Divergence Loss
    kl_loss = -0.5 * torch.sum(1 + z0_logvar - z0_mean.pow(2) - z0_logvar.exp())

    # Total Loss (ELBO)
    loss = recon_loss + kl_weight * kl_loss

    loss.backward()
    optimizer.step()

    history['total_loss'].append(loss.item())
    history['kl'].append(kl_loss.item())
    history['rec_loss'].append(recon_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}/{num_epochs}, Total Loss: {loss.item():.4f}, "
              f"Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")

print("Training finished.")

fig = plt.figure(figsize=(9,12))
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(np.arange(len(history['total_loss'])), history['total_loss'])
ax1.set_xlabel("Iteration")
ax1.set_title("Total Loss")

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(np.arange(len(history['kl'])), history['kl'])
ax2.set_xlabel("Iteration")
ax2.set_title("KL Loss")

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(np.arange(len(history['rec_loss'])), history['rec_loss'])
ax3.set_xlabel("Iteration")
ax3.set_title("Reconstruction Loss")

plt.grid()
plt.tight_layout()
plt.show()


# --- Visualization of Results ---
print("Generating final trajectory and plotting...")
model.eval()
with torch.no_grad():
    # We use the mean of the learned distribution for a deterministic trajectory
    z0_mean_final, _ = model.encoder(observations_batch)
    inferred_latent_trajectory = odeint(model.dynamics, z0_mean_final, t, method='dopri5')

# Convert to numpy for plotting
inferred_np = inferred_latent_trajectory.squeeze(1).cpu().numpy()
true_np = true_latent.cpu().numpy()

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(true_np[0, :], true_np[1, :], true_np[2, :], lw=1, color='blue')

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(inferred_np[:, 0], inferred_np[:, 1], inferred_np[:, 2], lw=1, color='red')

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
plt.show()



