import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Normal, kl_divergence

# Assuming 'src' directory is in the same folder or in the python path
from src.dataset import SyntheticDataGenerator
from src.models.vrnn.model import VRNN, calculate_loss

# Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_samples = 500
noise_variance = 0.1
random_seed = 42
MODEL_SAVE_PATH = 'vrnn_model_weights.pth'  # Define a path for the saved weights

# --- Hyperparameters for the VRNN Model ---
latent_dim = 3
hidden_dim = 64
obs_dim = 10
num_epochs = 2000
learning_rate = 1e-3

# --- Generate Data ---
print(f"Using device: {device}")
gen = SyntheticDataGenerator(
    num_steps=num_samples,
    time_step=1 / 2,
    noise_variance=noise_variance,
    random_seed=random_seed,
    device=device
)

synthetic_data = gen.generate_lorenz_3d()
observation_z = synthetic_data["observation_z"]  # Shape: (500, 10)
true_latent = synthetic_data["latent"]  # Shape: (500, 3)

# Reshape data for RNN: (seq_len, batch_size, feature_dim)
observations_batch = observation_z.unsqueeze(1)  # Shape: (500, 1, 10)

# Plot the ground truth latent dynamics
print("Displaying ground truth Lorenz attractor...")
gen.plot_lorenz_3d(true_latent)
plt.show()


# --- Training Loop ---
print("\nStarting VRNN training...")
model = VRNN(obs_dim, latent_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    reconstructions, prior_params, posterior_params = model(observations_batch)
    loss = calculate_loss(observations_batch, reconstructions, prior_params, posterior_params)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training finished.")

# --- 1. Saving the Model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model weights saved to {MODEL_SAVE_PATH}")

# --- 2. Loading the Model for Inference ---
# Create a new, untrained instance of the model
loaded_model = VRNN(obs_dim, latent_dim, hidden_dim).to(device)

# Load the saved weights from the file
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Set the model to evaluation mode (important for layers like dropout, batchnorm)
loaded_model.eval()
print("Model weights loaded successfully for inference.")

# --- Visualization of Results (using the loaded model) ---
print("Generating final latent trajectory from loaded model and plotting...")
with torch.no_grad():
    _, _, posterior_params_final = loaded_model(observations_batch)

# Extract the means of the posterior distributions to form the trajectory
inferred_latent_trajectory = torch.stack([p[0] for p in posterior_params_final]).squeeze(1)

# Convert to numpy for plotting
inferred_np = inferred_latent_trajectory.cpu().numpy()
true_np = true_latent.cpu().numpy()

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(true_np[:, 0], true_np[:, 1], true_np[:, 2], lw=1, color='blue')
ax1.set_title("Ground Truth Latent Dynamics")
ax1.set_xlabel("X");
ax1.set_ylabel("Y");
ax1.set_zlabel("Z")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(inferred_np[:, 0], inferred_np[:, 1], inferred_np[:, 2], lw=1, color='red')
ax2.set_title("Inferred Latent Dynamics (Loaded VRNN)")
ax2.set_xlabel("X");
ax2.set_ylabel("Y");
ax2.set_zlabel("Z")

plt.suptitle("Comparison of True vs. Inferred Latent Trajectories")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()