import torch
import torch.nn as nn
from torchdiffeq import odeint



# --- Step 1: Define the Model Components (Your code is perfect here) ---

# 1. Encoder (RNN) to infer the initial latent state z0
class Encoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(obs_dim, hidden_dim, batch_first=False)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, observations):
        # Process the first few points to infer initial state
        # observations shape: (seq_len, batch_size, obs_dim)
        _, h_last = self.rnn(observations)
        h_last = h_last.squeeze(0)
        mean = self.fc_mean(h_last)
        logvar = self.fc_logvar(h_last)
        return mean, logvar

# 2. Neural network to define the ODE's dynamics
class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=3):
        super(LatentDynamics, self).__init__()
        # Made the network a bit deeper to capture more complexity
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, latent_dim),
        )

    def forward(self, t, z):
        return self.net(z)

# 3. Decoder to map latent state back to observation space
class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim)
        )

    def forward(self, latent_z):
        return self.net(latent_z)

# 4. Main Latent ODE Model
class LatentODE(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super(LatentODE, self).__init__()
        self.encoder = Encoder(obs_dim, hidden_dim, latent_dim)
        self.dynamics = LatentDynamics(latent_dim)
        self.decoder = Decoder(latent_dim, obs_dim)

    def forward(self, observations, time_steps):
        # Use only a subset of initial points to encode z0
        num_encode_points = 50
        z0_mean, z0_logvar = self.encoder(observations[:num_encode_points, :, :])

        # Reparameterization trick
        std = torch.exp(0.5 * z0_logvar)
        epsilon = torch.randn_like(std)
        z0_sample = z0_mean + epsilon * std

        # z0_sample = torch.zeros(batch_size, self.dynamics.net[0].in_features).to(observations.device)

        # Solve the ODE
        latent_trajectory = odeint(self.dynamics, z0_sample, time_steps, method='dopri5')

        # Decode the latent trajectory
        reconstructed_obs = self.decoder(latent_trajectory)

        return reconstructed_obs, z0_mean, z0_logvar, latent_trajectory