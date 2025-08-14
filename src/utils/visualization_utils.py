from typing import Dict, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch


def plot_hisde_3d_matplotlib(
    Z, Y_fit, Y_true, Xs, Ms, dt, tau_idx,
    *,
    # heatmap
    cmap_heatmap="viridis",
    figsize_heatmap=(6.5, 4.2),
    # 3D
    color_by_time=True,
    linewidth_pred=1.8,
    linewidth_true=1.4,
    color_pred=None,      # if None and color_by_time=True, uses a time gradient
    color_true="tab:gray",
    elev=25.0,
    azim=-60.0,
    figsize_3d=(7.0, 5.5),
    show_event_connectors=True,
    # marks vs tau
    figsize_marks=(6.5, 3.6),
    marker_size=24,
    save_paths: dict | None = None,  # e.g., {"heatmap":"heat.png", "traj":"traj.png", "marks":"marks.png"}
    show=True,
):
    """
    Matplotlib-based replacement for the Plotly version:
      1) Heatmap of observed data Z
      2) 3D latent & predicted trajectories (with optional relative-time gradient)
      3) Marks vs event indices (tau)

    Args mirror your original function, with additional styling knobs.
    Returns:
        (fig_heatmap, ax_heatmap), (fig_traj, ax_traj3d), (fig_marks, ax_marks)
    """
    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    # --- Prepare data ---
    Z_np = to_np(Z)
    Y_true_np = to_np(Y_true)
    Yf = to_np(Y_fit)

    # Expect Yf shapes like (3, K, N) or (K, 3, N). We want mean over axis=1 (time dim if transposed below).
    if Yf.ndim == 3 and Yf.shape[0] != 3:
        # bring to (3, K, N)
        Yf = Yf.transpose(1, 0, 2)
    # mean over N (inducing/particles) to get (3, K)
    Y_fit_mean = Yf.mean(axis=2) if Yf.ndim == 3 else Yf
    if Y_fit_mean.ndim != 2 or Y_fit_mean.shape[0] != 3:
        raise ValueError("Y_fit should yield a (3, K, N) or (3, K) after processing.")

    # true latent expected as (3, K)
    if Y_true_np.ndim == 2 and Y_true_np.shape[0] == 3:
        pass
    elif Y_true_np.ndim == 2 and Y_true_np.shape[1] == 3:
        Y_true_np = Y_true_np.T
    else:
        raise ValueError("Y_true must have shape (3, K) or (K, 3).")

    # events along true latent at indices tau_idx
    P = Y_true_np[:, tau_idx]  # (3, n_events)

    # marks
    M_all = to_np(Ms)[0, 0, :]
    n_events = len(tau_idx)
    marks = M_all[:n_events]

    # ========================
    # 1) Heatmap of Z (matplotlib)
    # ========================
    fig_hm, ax_hm = plt.subplots(figsize=figsize_heatmap)
    im = ax_hm.imshow(Z_np, aspect="auto", origin="lower", cmap=cmap_heatmap)
    cbar = fig_hm.colorbar(im, ax=ax_hm, pad=0.02, fraction=0.04)
    cbar.set_label("Intensity")
    ax_hm.set_title("Observed data heatmap")
    ax_hm.set_xlabel("Time index")
    ax_hm.set_ylabel("Feature / Dimension")
    fig_hm.tight_layout()

    # ========================
    # 2) 3D latent & predicted (matplotlib)
    # ========================
    fig_3d = plt.figure(figsize=figsize_3d)
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    ax_3d.view_init(elev=elev, azim=azim)

    def _plot_time_gradient(ax3d, arr3xk, lw, label, color=None):
        """
        Plot a 3D line with optional time gradient (relative time).
        arr3xk: (3, K)
        """
        x, y, z = arr3xk[0], arr3xk[1], arr3xk[2]
        k = x.shape[0]
        if color_by_time and color is None and k > 1:
            pts = np.vstack([x, y, z]).T.reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            t = np.linspace(0.0, 1.0, k - 1)
            lc = Line3DCollection(segs, cmap="viridis", linewidth=lw)
            lc.set_array(t)
            ax3d.add_collection3d(lc)
            # autoscale
            ax3d.auto_scale_xyz(x, y, z)
            return lc  # so we can make a colorbar
        else:
            ax3d.plot(x, y, z, linewidth=lw, color=color, label=label)
            return None

    # Predicted mean (gradient if color_by_time and color_pred is None)
    lc_pred = _plot_time_gradient(ax_3d, Y_fit_mean, linewidth_pred, "Predicted", color=color_pred)
    # True latent (solid/dashed single color to contrast with gradient)
    ax_3d.plot(
        Y_true_np[0], Y_true_np[1], Y_true_np[2],
        linewidth=linewidth_true, color=color_true, linestyle="--", label="Latent (true)"
    )

    # Start/end markers for orientation
    ax_3d.scatter([Y_fit_mean[0, 0]], [Y_fit_mean[1, 0]], [Y_fit_mean[2, 0]], s=25, label="pred start")
    ax_3d.scatter([Y_fit_mean[0, -1]], [Y_fit_mean[1, -1]], [Y_fit_mean[2, -1]], s=25, label="pred end")

    # Event markers sampled from true
    ax_3d.scatter(P[0], P[1], P[2], s=18, color="red", label="Events")

    # Optional connectors between consecutive event points (on true path)
    if show_event_connectors and P.shape[1] > 1:
        for i in range(P.shape[1] - 1):
            ax_3d.plot(
                [P[0, i], P[0, i + 1]],
                [P[1, i], P[1, i + 1]],
                [P[2, i], P[2, i + 1]],
                color="black", linewidth=1.0, alpha=0.7
            )

    # Labels & legend
    ax_3d.set_xlabel("Y₁")
    ax_3d.set_ylabel("Y₂")
    ax_3d.set_zlabel("Y₃")
    ax_3d.set_title("3D latent (rela) and predicted trajectories")

    # If we used a gradient for predicted, add a colorbar for relative time
    if lc_pred is not None:
        cbar3d = fig_3d.colorbar(lc_pred, ax=ax_3d, pad=0.1, fraction=0.03)
        cbar3d.set_label("Relative time")

    ax_3d.legend(loc="upper left")
    fig_3d.tight_layout()

    # ========================
    # 3) Marks vs tau (matplotlib)
    # ========================
    fig_mk, ax_mk = plt.subplots(figsize=figsize_marks)
    ax_mk.plot(tau_idx, marks, marker="o", linewidth=1.2, markersize=np.sqrt(marker_size))
    ax_mk.set_title("Mark vs. τ")
    ax_mk.set_xlabel("Event time index (τ)")
    ax_mk.set_ylabel("Mark value (dim 1)")
    ax_mk.grid(True, linewidth=0.4, alpha=0.5)
    fig_mk.tight_layout()

    # ----- optional saving -----
    if save_paths:
        if "heatmap" in save_paths and save_paths["heatmap"]:
            fig_hm.savefig(save_paths["heatmap"], bbox_inches="tight", dpi=200)
        if "traj" in save_paths and save_paths["traj"]:
            fig_3d.savefig(save_paths["traj"], bbox_inches="tight", dpi=200)
        if "marks" in save_paths and save_paths["marks"]:
            fig_mk.savefig(save_paths["marks"], bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return (fig_hm, ax_hm), (fig_3d, ax_3d), (fig_mk, ax_mk)


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
