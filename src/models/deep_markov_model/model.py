import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Normal, kl_divergence

# Assuming 'src' directory is in the same folder or in the python path
from src.dataset import SyntheticDataGenerator

# --- DMM Model Definition ---
class DMM(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super(DMM, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Inference network (processes observations)
        self.encoder_rnn = nn.GRU(obs_dim, hidden_dim, batch_first=False)

        # Transition network (prior): p(z_t | z_{t-1})
        self.transition_net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, latent_dim * 2))

        # Posterior network q(z_t | h_t, z_{t-1})
        self.posterior_net = nn.Sequential(nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, latent_dim * 2))

        # Decoder/Emission network: p(x_t | z_t)
        self.decoder_net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, obs_dim))

    def forward(self, x):
        seq_len, batch_size, _ = x.shape

        # Run the inference RNN over the entire sequence first
        rnn_hiddens, _ = self.encoder_rnn(x)

        # Initialize the first latent state (e.g., from a standard normal)
        z_prev = torch.zeros(batch_size, self.latent_dim, device=x.device)

        all_posterior_params = []
        all_prior_params = []
        all_reconstructions = []

        for t in range(seq_len):
            # 1. Calculate prior parameters from previous latent state z_{t-1}
            prior_params = self.transition_net(z_prev)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

            # 2. Calculate posterior parameters by combining prior and evidence from data (h_t)
            posterior_input = torch.cat([rnn_hiddens[t], z_prev], dim=1)
            posterior_params = self.posterior_net(posterior_input)
            posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)

            # 3. Sample z_t from posterior using reparameterization trick
            std = torch.exp(0.5 * posterior_logvar)
            eps = torch.randn_like(std)
            z_t = posterior_mean + eps * std

            # 4. Reconstruct x_t from z_t
            x_recon = self.decoder_net(z_t)

            # Update previous latent state for the next step
            z_prev = z_t

            # Store results
            all_posterior_params.append((posterior_mean, posterior_logvar))
            all_prior_params.append((prior_mean, prior_logvar))
            all_reconstructions.append(x_recon)

        return all_reconstructions, all_prior_params, all_posterior_params


def calculate_loss(x, reconstructions, prior_params, posterior_params):
    """This loss function is identical to the one used for the VRNN."""
    seq_len = x.shape[0]
    recon_loss = 0.0
    kl_loss = 0.0

    for t in range(seq_len):
        # Reconstruction loss (MSE)
        recon_loss += nn.functional.mse_loss(reconstructions[t], x[t], reduction='sum')

        # KL divergence between posterior q(z_t|...) and prior p(z_t|z_{t-1})
        post_mean, post_logvar = posterior_params[t]
        prior_mean, prior_logvar = prior_params[t]

        q_dist = Normal(post_mean, torch.exp(0.5 * post_logvar))
        p_dist = Normal(prior_mean, torch.exp(0.5 * prior_logvar))

        kl_loss += kl_divergence(q_dist, p_dist).sum()

    return (recon_loss + kl_loss) / (seq_len * x.shape[1])