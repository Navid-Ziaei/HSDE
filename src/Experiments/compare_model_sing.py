import torch
from matplotlib import pyplot as plt

from src.dataset import SyntheticDataGenerator
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jr

from src.models.sing.sde import LinearSDE
from src.models.sing.likelihoods import Gaussian
from src.models.sing.sing import fit_variational_em

# Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mode = '3d'
duration_sec = 250
sampling_rate_hz = 2
num_samples = int(duration_sec * sampling_rate_hz)
time_step = 1 / sampling_rate_hz
noise_variance = 0.1
random_seed = 42


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
ys = observation_z.unsqueeze(0)  # Shape: (1, 500, 10)


# Shapes
n_trials = 1
n_timesteps = num_samples
obs_dim = ys.shape[-1]  # 10
latent_dim = 3

# Convert PyTorch tensors -> NumPy -> JAX
ys_np = ys.squeeze(0).detach().cpu().numpy()           # (T, 10)
ys_jnp = jnp.asarray(ys_np[None, :, :])                # (1, T, 10) for SING
t_grid = jnp.asarray((jnp.arange(n_timesteps) * time_step))  # length T
t_mask = jnp.ones((n_trials, n_timesteps))

# Likelihood
likelihood = Gaussian(ys_jnp, t_mask)

# Model
fn = LinearSDE(latent_dim=latent_dim)

# Reasonable/stable initializations
key = jr.PRNGKey(0)
drift_params_init = {
    "A": -0.05 * jnp.eye(latent_dim),   # small, stable linear drift
    "b": jnp.zeros((latent_dim,))
}
init_params_sde = {
    "mu0": jnp.zeros((n_trials, latent_dim)),
    "V0": jnp.tile(jnp.eye(latent_dim), (n_trials, 1, 1))
}
output_params_init = {
    "C": 0.1 * jr.normal(key, (obs_dim, latent_dim)),
    "d": jnp.zeros((obs_dim,)),
    "R": 0.1 * jnp.ones((obs_dim,))
}

# Training schedule
n_iters = 3       # you can increase for better fits
n_iters_e = 1      # inner E steps
rho_sched = jnp.ones(n_iters)

# Fit
results_sing = fit_variational_em(
    key,
    fn,
    likelihood,
    t_grid,
    drift_params_init,
    init_params_sde,
    output_params_init,
    batch_size=None,
    rho_sched=rho_sched,
    n_iters=n_iters,
    n_iters_e=n_iters_e,
    perform_m_step=True,   # learn A,b,C,d,R
)

# Unpack results
marginal_params, natural_params, gp_post, drift_params, init_params_fitted, output_params_fitted, input_effect, elbos = results_sing
ms_final, Ss_final, _ = marginal_params  # ms_final: (1, T, 3), Ss_final: (1, T, 3, 3)

print("Final ELBO:", float(elbos[-1]))

# --- Plot: true vs inferred latents (if you kept the ground-truth from the generator) ---
# true_latent_torch is (3, 1, T) in your snippet; convert back to (T, 3) for plotting
true_latent_np = true_latent_torch.squeeze(1).t().detach().cpu().numpy().transpose()  # (T, 3)

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
names = [r"$x_1$", r"$x_2$", r"$x_3$"]

for i in range(latent_dim):
    axes[i].plot(t_numpy, true_latent_np[:, i], lw=1.5, label="true", zorder=1)
    axes[i].plot(t_numpy, ms_final[0, :, i], lw=1.5, label="inferred", zorder=2)
    axes[i].fill_between(
        t_numpy,
        ms_final[0, :, i] + 2 * jnp.sqrt(Ss_final[0, :, i, i]),
        ms_final[0, :, i] - 2 * jnp.sqrt(Ss_final[0, :, i, i]),
        alpha=0.3, zorder=0
    )
    axes[i].set_ylabel(names[i])

axes[-1].set_xlabel("time (s)")
axes[0].legend(loc="upper right")
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
ax1.plot(true_latent_np[:, 0], true_latent_np[:, 1], true_latent_np[:, 2], lw=1, color='blue')
ax1.plot(ms_final[0, :, 0], ms_final[0, :, 1], ms_final[0, :, 2], lw=1, color='red')

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

plt.show()