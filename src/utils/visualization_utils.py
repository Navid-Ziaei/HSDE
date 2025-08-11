from typing import Dict, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

# -----------------------------
# Curves: log-marginal, MSE, params over EM iters
# References:
#   - log_marginal ~ evidence via PF normalizers [Eq. \ref{eq17}]
#   - mse_observed from reconstruction Zhat = W E[Y] [Eq. \ref{eq6}]
#   - sigma_x (X-layer SDE noise), k_tau/lambda_tau (renewal priors) [Sec. \ref{a33}, Eq. (2)]
# -----------------------------
def plot_training_curves(curves: Dict[str, list], figsize=(12, 8)) -> plt.Figure:
    it = np.asarray(curves["iter"])
    log_marginal = np.asarray(curves["log_marginal"], dtype=float)
    mse_obs = np.asarray(curves["mse_observed"], dtype=float)
    sigma_x = np.asarray(curves["sigma_x"], dtype=float)
    k_tau = np.asarray(curves["k_tau"], dtype=float)
    lmb_tau = np.asarray(curves["lmb_tau"], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    ax = axes.ravel()

    ax[0].plot(it, log_marginal, marker='o')
    ax[0].set_title("PF log-marginal (↑ better)  [Eq. (17)]")
    ax[0].set_xlabel("EM iteration")
    ax[0].set_ylabel("log p(Z_{1:K})")

    ax[1].plot(it, mse_obs, marker='o')
    ax[1].set_title("Reconstruction MSE  [Eq. (6)]")
    ax[1].set_xlabel("EM iteration")
    ax[1].set_ylabel("MSE (Z vs W·E[Y])")

    ax[2].plot(it, sigma_x, marker='o')
    ax[2].set_title(r"$\sigma_x$  [Sec. A.33]")
    ax[2].set_xlabel("EM iteration")
    ax[2].set_ylabel("value")

    ax[3].plot(it, k_tau, marker='o')
    ax[3].set_title(r"Gamma shape $k_\tau$  [Eq. (2)]")
    ax[3].set_xlabel("EM iteration")
    ax[3].set_ylabel("value")

    ax[4].plot(it, lmb_tau, marker='o')
    ax[4].set_title(r"Gamma rate $\lambda_\tau$  [Eq. (2)]")
    ax[4].set_xlabel("EM iteration")
    ax[4].set_ylabel("value")

    # empty or add a legend/table as needed
    ax[5].axis('off')
    return fig


# -----------------------------
# Trajectories: inferred X, inferred Y, reconstructed Z with observed/clean
# and vertical lines for "typical" inducing points
# References:
#   - X/Y discrete dynamics [Sec. \ref{a33}]
#   - Observation model Z = W Y + eps [Eq. \ref{eq6}]
#   - Inducing intervals via Gamma [Eq. (2)]; events sketched via median tau from SMC samples [Alg. \ref{alg2}]
# -----------------------------
def plot_inference_and_inducing_points(
    model,
    Z: torch.Tensor,                           # (K, M)
    clean: Optional[torch.Tensor] = None,      # (K,) or (K,M)
    num_particles: Optional[int] = None,
    ess_fraction: float = 0.5,
    seed: Optional[int] = None,
    figsize=(14, 8)
) -> plt.Figure:
    device = next(model.parameters()).device
    Z = Z.to(device)

    # 1) Run inference requesting particles to compute posterior means per-step
    inf = model.infer(
        Z,
        num_particles=num_particles,
        ess_fraction=ess_fraction,
        seed=seed,
        return_particles=True,  # we’ll compute E[X_k], E[Y_k] explicitly
    )

    # E[Y_k] already provided
    EY = inf["EY"].detach().cpu().numpy()            # (K, D)
    Zhat = inf["Zhat"].detach().cpu().numpy()        # (K, M)

    # Compute E[X_k] from stored particle snapshots and weights at each k
    # particles_X: (K, U, D), particles_w: (K, U)
    EX = None
    if "particles_X" in inf and "particles_w" in inf:
        Xk = inf["particles_X"].detach().cpu()       # (K, U, D)
        Wk = inf["particles_w"].detach().cpu()       # (K, U)
        EX = torch.einsum("ku,kud->kd", Wk, Xk).numpy()  # (K, D)

    # 2) Build a set of "typical" inducing times from SMC tau-samples
    # (We use median tau and accumulate to T = K*dt)
    smc_out = model.smc(Z, num_particles=num_particles, ess_fraction=ess_fraction, seed=seed)
    tau_samples = smc_out.get("tau_samples", None)
    t_events = []
    if tau_samples is not None and tau_samples.numel() > 0:
        tau_med = float(tau_samples.median().cpu().item())
        T = model.K * model.dt
        t = tau_med
        while t < T:
            t_events.append(t)
            t += tau_med
    t_events = np.asarray(t_events, dtype=float)

    # 3) Prepare data for plotting
    K, M = Z.shape
    t_grid = (torch.arange(K, dtype=torch.float32) * model.dt).cpu().numpy()

    Z_np = Z.detach().cpu().numpy()
    if clean is not None:
        clean = clean.to(device)
        if clean.ndim == 1:
            clean_np = clean.detach().cpu().numpy()[:, None]  # (K,1)
        else:
            clean_np = clean.detach().cpu().numpy()           # (K,M)
    else:
        clean_np = None

    # D and M might be >1; for visualization we take the first dim for X, Y, Z
    d0 = 0
    m0 = 0

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)

    # --- (1) inferred X ---
    ax = axes[0]
    if EX is not None:
        ax.plot(t_grid, EX[:, d0])
    else:
        ax.text(0.02, 0.85, "E[X] not stored; re-run infer(return_particles=True)", transform=ax.transAxes)
    ax.set_ylabel(r"$\mathbb{E}[X_k]$  [Sec. A.33]")
    ax.set_title("Inferred X, Inferred Y, and Reconstructed Z with Inducing Points")

    # vertical event lines
    for te in t_events:
        ax.axvline(te, linestyle="--", alpha=0.4)

    # --- (2) inferred Y ---
    ax = axes[1]
    ax.plot(t_grid, EY[:, d0])
    ax.set_ylabel(r"$\mathbb{E}[Y_k]$  [Sec. A.33]")
    for te in t_events:
        ax.axvline(te, linestyle="--", alpha=0.4)

    # --- (3) reconstruction vs observed (and clean) ---
    ax = axes[2]
    ax.plot(t_grid, Z_np[:, m0], label="Observed Z")
    ax.plot(t_grid, Zhat[:, m0], label=r"Reconstruction $\hat Z = W\,\mathbb{E}[Y]$  [Eq. (6)]")
    if clean_np is not None:
        ax.plot(t_grid, clean_np[:, m0], label="Clean target")
    for te in t_events:
        ax.axvline(te, linestyle="--", alpha=0.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Z (dim 0)")
    ax.legend(loc="best")

    return fig
