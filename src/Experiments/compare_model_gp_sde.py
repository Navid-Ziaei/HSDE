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

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt



from src.models.gpsde.inference import GaussMarkovLagrange
from src.models.gpsde.likelihoods import Gaussian
from src.models.gpsde.mappings import AffineMapping
from src.models.gpsde.transition import SparseGP  # start simple; you can swap to FixedPointSparseGP later
from src.models.gpsde.kernels import RBF
from src.models.gpsde.models import GPSDEmodel, GPSDE



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
true_latent = synthetic_data["latent"].t()  # Shape: (500, 3)
true_latent_torch = true_latent.to(device).t().unsqueeze(1)

# Reshape data for RNN: (seq_len, batch_size, feature_dim)
# We treat the entire time series as a single batch item.
ys = observation_z.unsqueeze(0)  # Shape: (1, 500, 10)


# GPSDE uses lists: Y[n] is (T_n, yDim) and tObs[n] is (T_n,)
yDim = observation_z.shape[1]
T = observation_z.shape[0]
assert T == len(t_numpy) == num_samples, "Time/obs length mismatch."

# single trial
Y = [observation_z.cpu().numpy(), observation_z.cpu().numpy()]            # (T, 10)
tObs = [t_numpy]                              # (T,)

# Inference will predict on a dense test grid (use same grid here)
testTimes = torch.from_numpy(t_numpy).to(torch.float64)

# --- Initialize affine observation mapping y = C x + d + noise ---
# Good init: least-squares fit from true latent -> observations.
# (Uses ground truth only for initialization; GPSDE will refine.)
X_true = true_latent.cpu().numpy()            # (T, 3)
Y_np = Y[0]                                   # (T, 10)

# Add bias and solve [X 1] w ≈ Y  => C,d via lstsq
X_aug = np.concatenate([X_true, np.ones((T, 1))], axis=1)   # (T, 4)
W, *_ = np.linalg.lstsq(X_aug, Y_np, rcond=None)            # (4, 10)
C_init = W[:3, :].T                                          # (10, 3)
d_init = W[3:, :].T                                          # (10,)

# Build mapping (expects shapes: C: (xDim, yDim), d: (1, yDim))
xDim = 3
C_t = torch.tensor(C_init, dtype=torch.float64)              # (10,3)
d_t = torch.tensor(d_init, dtype=torch.float64)              # (10,)
outputMapping = AffineMapping(C_t.T.contiguous(), d_t.view(1, -1), useClosedForm=True)

# --- Likelihood (Gaussian, diagonal R) ---
R0 = 0.5 * torch.ones(yDim, 1, dtype=torch.float64)          # diagonal obs noise init
like = Gaussian(Y, tObs, np.array([duration_sec]), R0, dtstep=time_step, useClosedForm=True)

# --- GP drift: RBF kernel in 3D with a small 3D inducing-grid ---
# Keep it modest to avoid heavy computation: 4 x 4 x 4 = 64 inducing points
lens = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64).reshape(-1, 1)  # per-dim lengthscales
kern = RBF(xDim, lens)

# Build a 3D grid in latent space (rough Lorenz ranges are ~[-25, 25] per coord; we start tighter)
grid_min, grid_max, grid_pts = -15.0, 15.0, 4
g = torch.linspace(grid_min, grid_max, grid_pts, dtype=torch.float64)
Zx, Zy, Zz = torch.meshgrid(g, g, g, indexing="ij")
Zs = torch.stack([Zx.reshape(-1), Zy.reshape(-1), Zz.reshape(-1)], dim=1)  # (64, 3)

transfunc = SparseGP(kern, Zs)  # vanilla sparse GP drift; you can try FixedPointSparseGP later

# --- Variational inference over Markov Gaussian posterior ---
inference = GaussMarkovLagrange(xDim, np.array([duration_sec]), learningRate=1.0, dtstep=time_step)

# --- Assemble GPSDE model ---
model = GPSDEmodel(xDim, transfunc, outputMapping, like, nLeg=100)
myGPSDE = GPSDE(model, inference)

# Optional: fix inducing points on the grid
myGPSDE.model.transfunc.Zs.requires_grad = False

# --- Train via variational EM ---
# You can increase iterations once it runs end-to-end (e.g., niter=10..50)
myGPSDE.variationalEM(niter=3, eStepIter=10, mStepIter=10)

# --- Infer latent marginals on the full time grid (trial idx = 0) ---
idx = 0
m, S = myGPSDE.inference.predict_marginals(idx, testTimes)  # m: (T, xDim, 1), S: (T, xDim, xDim)

# --- Plot: inferred vs. true latents ---
true_latent_np = true_latent.cpu().numpy()  # (T, 3)

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
labels = [r"$x_1(t)$", r"$x_2(t)$", r"$x_3(t)$"]

for i in range(xDim):
    axes[i].plot(t_numpy, true_latent_np[:, i], linewidth=1.0, label="true")
    axes[i].plot(t_numpy, m[:, i, 0].detach().numpy(), linewidth=1.0, label="inferred")
    axes[i].fill_between(
        t_numpy,
        (m[:, i, 0] + torch.sqrt(torch.clamp(S[:, i, i], min=1e-9))).detach().numpy(),
        (m[:, i, 0] - torch.sqrt(torch.clamp(S[:, i, i], min=1e-9))).detach().numpy(),
        alpha=0.25,
    )
    axes[i].set_ylabel(labels[i])

axes[-1].set_xlabel("time (s)")
axes[0].legend(loc="upper right")
plt.tight_layout()
plt.show()