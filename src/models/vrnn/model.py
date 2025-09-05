import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence



# --- VRNN Model Definition ---
class VRNN(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super(VRNN, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Feature extractors
        self.phi_x = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.phi_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())

        # Prior network: p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, latent_dim * 2))

        # Encoder network: q(z_t | x_t, h_{t-1})
        self.encoder_net = nn.Sequential(nn.Linear(hidden_dim + hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, latent_dim * 2))

        # Decoder network: p(x_t | z_t, h_{t-1})
        self.decoder_net = nn.Sequential(nn.Linear(hidden_dim + hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, obs_dim))

        # Recurrent layer
        self.rnn = nn.GRUCell(hidden_dim + hidden_dim, hidden_dim)

    def forward(self, x):
        seq_len, batch_size, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        all_posterior_params = []
        all_prior_params = []
        all_reconstructions = []

        for t in range(seq_len):
            x_t = x[t, :, :]
            phi_x_t = self.phi_x(x_t)

            prior_params = self.prior_net(h)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

            encoder_input = torch.cat([phi_x_t, h], dim=1)
            posterior_params = self.encoder_net(encoder_input)
            posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)

            std = torch.exp(0.5 * posterior_logvar)
            eps = torch.randn_like(std)
            z_t = posterior_mean + eps * std
            phi_z_t = self.phi_z(z_t)

            decoder_input = torch.cat([phi_z_t, h], dim=1)
            x_recon = self.decoder_net(decoder_input)

            rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1)
            h = self.rnn(rnn_input, h)

            all_posterior_params.append((posterior_mean, posterior_logvar))
            all_prior_params.append((prior_mean, prior_logvar))
            all_reconstructions.append(x_recon)

        return all_reconstructions, all_prior_params, all_posterior_params


def calculate_loss(x, reconstructions, prior_params, posterior_params):
    seq_len = x.shape[0]
    recon_loss = 0.0
    kl_loss = 0.0

    for t in range(seq_len):
        recon_loss += nn.functional.mse_loss(reconstructions[t], x[t], reduction='sum')
        post_mean, post_logvar = posterior_params[t]
        prior_mean, prior_logvar = prior_params[t]
        q_dist = Normal(post_mean, torch.exp(0.5 * post_logvar))
        p_dist = Normal(prior_mean, torch.exp(0.5 * prior_logvar))
        kl_loss += kl_divergence(q_dist, p_dist).sum()

    return (recon_loss + kl_loss) / (seq_len * x.shape[1])