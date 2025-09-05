import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'src' directory is in the python path
from src.dataset import SyntheticDataGenerator

# --- Try to import Mamba ---
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba is not installed. Please run 'pip install causal-conv1d mamba-ssm'")
    Mamba = None  # Set to None if import fails

# Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_samples = 500
noise_variance = 0.1
random_seed = 42

# --- Hyperparameters ---
latent_dim = 3
d_model = 64  # Hidden dimension for S4/Mamba
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

# Reshape data for S4/Mamba: (batch_size, seq_len, feature_dim)
observations_batch = observation_z.unsqueeze(0)  # Shape: (1, 500, 10)

# Plot the ground truth latent dynamics
print("Displaying ground truth Lorenz attractor...")
gen.plot_lorenz_3d(true_latent)
plt.show()


# --- Model Definitions ---

# NOTE: The official S4 library is separate. Mamba's architecture is an evolution of
# S4's core principles (structured state space models). We will use the Mamba block
# here as it's the modern, high-performance successor. For simplicity,
# we will create one model class that uses the Mamba block.
# If you wish to use a pure S4 model, you would install the 's4' library
# and replace the `Mamba` layer with an `S4` layer.

class MambaAutoencoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, d_model, d_state=16, d_conv=4):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba is not installed.")

        self.encoder = nn.Linear(obs_dim, d_model)

        # Mamba layer as the core sequence processor
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )

        self.decoder = nn.Linear(d_model, obs_dim)
        self.latent_projector = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, obs_dim)

        # 1. Encode observations to d_model
        encoded_x = self.encoder(x)

        # 2. Process sequence with Mamba
        hidden_states = self.mamba(encoded_x)

        # 3. Decode back to observation space
        reconstructions = self.decoder(hidden_states)

        # 4. Project hidden states to 3D for visualization
        inferred_latents = self.latent_projector(hidden_states)

        return reconstructions, inferred_latents


# --- Generic Training and Visualization Function ---
def train_and_visualize(model, model_name, train_data, true_latents):
    print(f"\n--- Training {model_name} Model ---")
    MODEL_SAVE_PATH = f'{model_name.lower()}_model_weights.pth'

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        reconstructions, _ = model(train_data)
        loss = criterion(reconstructions, train_data)

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item():.6f}")

    print("Training finished.")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model weights saved to {MODEL_SAVE_PATH}")

    # --- Visualization ---
    model.eval()
    with torch.no_grad():
        _, inferred_latents = model(train_data)

    inferred_np = inferred_latents.squeeze(0).cpu().numpy()
    true_np = true_latents.cpu().numpy()

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(true_np[:, 0], true_np[:, 1], true_np[:, 2], lw=1, color='blue')
    ax1.set_title("Ground Truth Latent Dynamics")
    ax1.set_xlabel("X");
    ax1.set_ylabel("Y");
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(inferred_np[:, 0], inferred_np[:, 1], inferred_np[:, 2], lw=1, color='green')
    ax2.set_title(f"Inferred Latent Dynamics ({model_name})")
    ax2.set_xlabel("X");
    ax2.set_ylabel("Y");
    ax2.set_zlabel("Z")

    plt.suptitle(f"Comparison using {model_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# --- Run the Experiment ---
if Mamba is not None:
    # Instantiate the Mamba-based model
    mamba_model = MambaAutoencoder(obs_dim, latent_dim, d_model).to(device)

    # Train and visualize
    train_and_visualize(
        model=mamba_model,
        model_name="Mamba",
        train_data=observations_batch,
        true_latents=true_latent
    )
else:
    print("Skipping experiment because Mamba is not installed.")