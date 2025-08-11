import plotly.graph_objects as go

import torch
from typing import Dict, Any, Tuple
import math
from scipy.optimize import minimize

from scipy.integrate import solve_ivp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def simulate_lorenz(tspan, dt, y0, sigma=10.0, beta=8 / 3, rho=14.0):
    """
    Simulate the Lorenz system and interpolate onto a uniform grid.

    Args:
        tspan (tuple of float): (t_start, t_end)
        dt (float): desired time-step for uniform output
        y0 (array-like of length 3): initial state [x0, y0, z0]
        sigma (float, optional): Lorenz σ parameter (default=10)
        beta  (float, optional): Lorenz β parameter (default=8/3)
        rho   (float, optional): Lorenz ρ parameter (default=14)

    Returns:
        t_uniform (np.ndarray of shape (M,)): uniform time points
        y_uniform (np.ndarray of shape (M, 3)): [x,y,z] state at t_uniform
    """

    def lorenz_ode(t, y):
        dx = sigma * (y[1] - y[0])
        dy = y[0] * (rho - y[2]) - y[1]
        dz = y[0] * y[1] - beta * y[2]
        return [dx, dy, dz]

    # 1) Integrate on an adaptive grid
    sol = solve_ivp(lorenz_ode, (tspan[0], tspan[1]), y0, method='RK45')
    t_sol = sol.t  # nonuniform time points
    y_sol = sol.y.T  # shape (len(t_sol), 3)

    # 2) Build uniform time grid
    t_uniform = np.arange(tspan[0], tspan[1] + dt, dt)
    t_uniform = t_uniform[t_uniform <= tspan[1]]

    # 3) Linearly interpolate each dimension
    y_uniform = np.empty((t_uniform.size, 3), dtype=float)
    for dim in range(3):
        y_uniform[:, dim] = np.interp(t_uniform, t_sol, y_sol[:, dim])

    return t_uniform, y_uniform


class HiSDE3D:
    def __init__(self, device: str = 'cpu'):
        # Device for all tensors
        self.device = torch.device(device)

        # 1) Event‐count and Gamma prior hyperparameters
        self.max_events = 100
        self.gam_a0 = 30.0  # prior shape α
        self.gam_b0 = 2.0  # prior rate (for α)
        self.gam_c0 = 5.0  # prior shape (for scale θ)
        self.gam_d0 = 6.0  # prior rate (for θ)

        # 2) Observation dimension and NIW hyperparameters for 3D state
        self.N = 10  # number of observed channels
        self.norm_mu0 = torch.zeros(3, device=self.device)  # prior mean for X(0), Y(0), marks
        self.norm_k0 = 1.0  # strength of prior mean
        self.norm_phi = 4.0 * torch.eye(3, device=self.device)  # prior scale matrix (3×3)
        self.norm_v0 = 5.0  # degrees of freedom (≥3+1)

        # 3) Simulation / SMC settings
        self.K = 500  # time‐series length
        self.dt = 0.5  # time‐step size
        self.sample = 20000  # number of particles

        # 4) Noise variances and measurement mapping
        self.x_var = 10 * 1e-1  # latent increment noise var σₓ²
        self.y_var = 1e-3  # integrator noise var σ_y²

        # Observation noise covariance: random diagonal (N×N)
        diag_elems = 0.1 * torch.abs(torch.randn(self.N, device=self.device))
        self.z_var = torch.diag(diag_elems)

        # Random measurement matrix (N×3)
        self.W = 0.1 * torch.randn(self.N, 3, device=self.device)

        # 5) Initialize EM‐estimated parameters
        self.mark_mean = torch.zeros(self.max_events, 3, device=self.device)
        self.mark_var = torch.zeros(self.max_events, 3, 3, device=self.device)
        self.gam_alpha = torch.zeros(self.max_events, device=self.device)
        self.gam_beta = torch.zeros(self.max_events, device=self.device)
        self.event_cnt = 0

        # 6) Scale adjustment for integrator update
        self.scale_adj = 1.0

    def smc(self, iteration: int, Z: torch.Tensor, cur_param) -> Dict[str, torch.Tensor]:
        """
        Run 3D Sequential Monte Carlo for the HiSDE model.
        """
        device = self.device
        D = 3
        N = self.N
        sample = self.sample
        dt = self.dt
        K = Z.size(0)
        scale_adj = self.scale_adj
        x_var, y_var = self.x_var, self.y_var

        # Gamma‐prior for waiting times
        gam_alpha_prior = self.gam_a0 / self.gam_b0
        gam_beta_prior = self.gam_c0 / self.gam_d0

        # NIW hyperpriors
        norm_mu0, norm_phi = self.norm_mu0, self.norm_phi

        # Measurement parameters
        W = cur_param.W  # [N×3]
        z_var = cur_param.z_var  # [N×N] diagonal covariance

        # Precompute multivariate‐Gaussian normalization constant and floor epsilon
        diag_z = torch.diagonal(z_var)  # [N]
        inv_diag = 1.0 / diag_z  # [N]
        det_z = torch.prod(diag_z)  # scalar |Σ|
        norm_const = (2 * math.pi) ** (-N / 2) * det_z ** (-0.5)
        eps = 10 * torch.finfo(z_var.dtype).tiny

        # Allocate storage
        Xs = torch.zeros(D, sample, K + 1, device=device)
        Ys = torch.zeros(D, sample, K + 1, device=device)
        Ms = torch.zeros(D, sample, self.max_events, device=device)
        Ts = torch.zeros(sample, self.max_events, device=device)
        Cntr = torch.ones(sample, dtype=torch.long, device=device)

        # INITIALIZE at t=0
        if iteration == 1:
            mvn0 = torch.distributions.MultivariateNormal(norm_mu0, covariance_matrix=norm_phi)
            x0 = mvn0.sample((sample,))  # [sample × 3]
            y0 = mvn0.sample((sample,))
        else:
            mvn_x = torch.distributions.MultivariateNormal(cur_param.x_mu0, covariance_matrix=cur_param.x_var0)
            mvn_y = torch.distributions.MultivariateNormal(cur_param.y_mu0, covariance_matrix=cur_param.y_var0)
            x0 = mvn_x.sample((sample,))  # [sample × 3]
            y0 = mvn_y.sample((sample,))

        Xs[:, :, 0] = x0.T
        Ys[:, :, 0] = y0.T

        # MAIN PARTICLE LOOP
        for k in tqdm(range(1, K + 1), desc = "Sequential update over k = 0...K-1"):
            t_k = k * dt
            weights = torch.zeros(sample, device=device)

            for s in range(sample):
                cnt = Cntr[s].item()
                last_t = Ts[s, cnt - 1]

                # PROPOSE new event if time passes last event
                if last_t < t_k:
                    if cnt <= cur_param.event_cnt:
                        a, b = cur_param.gam_alpha[cnt - 1], cur_param.gam_beta[cnt - 1]
                    else:
                        a, b = gam_alpha_prior, gam_beta_prior

                    gamma_dist = torch.distributions.Gamma(concentration=a, rate=1.0 / b)
                    wait = torch.clamp(gamma_dist.sample(), min=2 * dt)
                    Ts[s, cnt] = last_t + wait

                    # sample 3D mark
                    if cnt <= cur_param.event_cnt:
                        mu_m, cov_m = cur_param.mark_mean[cnt - 1], cur_param.mark_var[cnt - 1]
                    else:
                        mu_m, cov_m = norm_mu0, norm_phi

                    mvn_m = torch.distributions.MultivariateNormal(mu_m, covariance_matrix=cov_m)
                    Ms[:, s, cnt] = mvn_m.sample()
                    Cntr[s] = cnt + 1

                # PROPAGATE X and Y
                idx = Cntr[s].item() - 1
                t2 = Ts[s, idx]
                t1 = Ts[s, idx - 1] if idx > 0 else torch.tensor(0.0, device=device)

                denom = torch.clamp(t2 - t_k, min=0.5 * dt)
                ak = 1.0 - dt / denom
                bk = Ms[:, s, idx] * (dt / denom)
                sk = (t2 - t_k) * (t_k - t1) / (t2 - t1 + 1e-8)

                noise_x = torch.randn(D, device=device) * math.sqrt(x_var * sk * dt)
                Xs[:, s, k] = ak * Xs[:, s, k - 1] + bk + noise_x

                noise_y = torch.randn(D, device=device) * math.sqrt(y_var * dt)
                Ys[:, s, k] = Ys[:, s, k - 1] + Xs[:, s, k - 1] * dt * scale_adj + noise_y

                # WEIGHT = mvnpdf(Z[k-1], W @ Ys[:,s,k], z_var)
                z_pred = W @ Ys[:, s, k]  # [N]
                diff = Z[k - 1] - z_pred  # [N]
                exponent = -0.5 * (diff ** 2 * inv_diag).sum()
                w = norm_const * torch.exp(exponent)
                # floor to avoid zero
                weights[s] = w.clamp(min=eps)

            # normalize and resample
            weights = weights / weights.sum()
            idxs = torch.multinomial(weights, sample, replacement=True)

            Ts = Ts[idxs]
            Cntr = Cntr[idxs]
            Xs = Xs[:, idxs, :]
            Ys = Ys[:, idxs, :]
            Ms = Ms[:, idxs, :]

        return {'Xs': Xs, 'Ys': Ys, 'Ms': Ms, 'Ts': Ts, 'Cntr': Cntr}

    def smc_fast(self, iteration: int, Z: torch.Tensor, cur_param) -> Dict[str, torch.Tensor]:
        """
        Vectorized over particles (sample dimension). Time loop remains sequential.
        Z: [K, N]
        """
        device = self.device
        D, N = 3, self.N
        sample = self.sample
        dt = self.dt
        K = Z.size(0)
        scale_adj = self.scale_adj
        x_var, y_var = self.x_var, self.y_var

        # Waiting-time priors
        gam_alpha_prior = self.gam_a0 / self.gam_b0
        gam_beta_prior = self.gam_c0 / self.gam_d0

        # NIW hyperpriors for prior marks
        norm_mu0, norm_phi = self.norm_mu0, self.norm_phi

        # Measurement params
        W = cur_param.W  # [N,3]
        z_var = cur_param.z_var  # [N,N] (diag)
        diag_z = torch.diagonal(z_var)  # [N]
        inv_diag = 1.0 / diag_z  # [N]
        det_z = torch.prod(diag_z)  # scalar
        norm_const = (2.0 * math.pi) ** (-N / 2) * det_z ** (-0.5)
        tiny = 1e-12

        # Allocate
        Xs = torch.zeros(D, sample, K + 1, device=device)
        Ys = torch.zeros(D, sample, K + 1, device=device)
        Ms = torch.zeros(D, sample, self.max_events, device=device)
        Ts = torch.zeros(sample, self.max_events, device=device)
        Cntr = torch.ones(sample, dtype=torch.long, device=device)

        # t=0 init (kept identical to your logic)
        if iteration == 1:
            mvn0 = torch.distributions.MultivariateNormal(norm_mu0, covariance_matrix=norm_phi)
            x0 = mvn0.sample((sample,))  # [sample,3]
            y0 = mvn0.sample((sample,))
        else:
            mvn_x = torch.distributions.MultivariateNormal(cur_param.x_mu0, covariance_matrix=cur_param.x_var0)
            mvn_y = torch.distributions.MultivariateNormal(cur_param.y_mu0, covariance_matrix=cur_param.y_var0)
            x0 = mvn_x.sample((sample,))
            y0 = mvn_y.sample((sample,))
        Xs[:, :, 0] = x0.T
        Ys[:, :, 0] = y0.T

        rows = torch.arange(sample, device=device)
        half_dt = 0.5 * dt
        sqrt_y_dt = math.sqrt(y_var * dt)

        # Precompute Cholesky for prior mark covariance
        L_prior = torch.linalg.cholesky(norm_phi + 1e-6 * torch.eye(3, device=device))  # [3,3]

        for k in tqdm(range(1, K + 1), desc="Sequential update over k = 0...K-1:"):
            t_k = k * dt

            # ---------------- 1) Event proposals (vectorized) ----------------
            last_idx = Cntr - 1  # [sample]
            t_last = Ts[rows, last_idx]  # [sample]
            event_mask = t_last < t_k  # [sample]

            if event_mask.any():
                rows_ev = rows[event_mask]  # [M]
                cnt_ev = Cntr[event_mask]  # [M]
                last_t_ev = Ts[rows_ev, cnt_ev - 1]  # [M]

                # pick Gamma params (known vs. prior)
                a_ev = torch.full_like(cnt_ev, gam_alpha_prior, dtype=torch.float32)
                b_ev = torch.full_like(cnt_ev, gam_beta_prior, dtype=torch.float32)

                if cur_param.event_cnt > 0:
                    known_mask = cnt_ev <= cur_param.event_cnt
                else:
                    known_mask = torch.zeros_like(cnt_ev, dtype=torch.bool)

                if known_mask.any():
                    idx_known_ev = cnt_ev[known_mask] - 1  # [M_known]
                    a_ev[known_mask] = cur_param.gam_alpha[idx_known_ev]
                    b_ev[known_mask] = cur_param.gam_beta[idx_known_ev]

                # sample waiting times
                gamma_ev = torch.distributions.Gamma(concentration=a_ev, rate=1.0 / b_ev)
                wait_ev = torch.clamp(gamma_ev.sample(), min=2.0 * dt)  # [M]
                Ts[rows_ev, cnt_ev] = last_t_ev + wait_ev

                # sample 3D marks at the new slot (cnt_ev)
                # Known subset
                if known_mask.any():
                    idx_e = cnt_ev[known_mask] - 1  # [M_known]
                    mu_k = cur_param.mark_mean[idx_e]  # [M_known,3]
                    cov_k = cur_param.mark_var[idx_e] + 1e-6 * torch.eye(3, device=device)  # [M_known,3,3]
                    mvn_k = torch.distributions.MultivariateNormal(mu_k, covariance_matrix=cov_k)
                    m_samp = mvn_k.sample()  # [M_known,3]
                    Ms[:, rows_ev[known_mask], cnt_ev[known_mask]] = m_samp.T

                # Prior subset
                prior_mask = ~known_mask
                if prior_mask.any():
                    M_prior = prior_mask.sum().item()
                    mu_p = norm_mu0.expand(M_prior, -1)  # [M_prior,3]
                    cov_p = norm_phi.expand(M_prior, -1, -1) + 1e-6 * torch.eye(3, device=device)
                    mvn_p = torch.distributions.MultivariateNormal(mu_p, covariance_matrix=cov_p)
                    m_samp = mvn_p.sample()  # [M_prior,3]
                    Ms[:, rows_ev[prior_mask], cnt_ev[prior_mask]] = m_samp.T

                # increment counters
                Cntr[event_mask] = cnt_ev + 1

            # ---------------- 2) State propagation (vectorized) ----------------
            last_idx = Cntr - 1
            t2 = Ts[rows, last_idx]  # [sample]

            has_prev = Cntr > 1
            t1 = torch.zeros(sample, device=device)
            t1[has_prev] = Ts[rows[has_prev], Cntr[has_prev] - 2]  # [sample]

            b_val = Ms[:, rows, last_idx]  # [3, sample]

            denom = torch.clamp(t2 - t_k, min=half_dt)  # [sample]
            ak = 1.0 - dt / denom  # [sample]
            r = dt / denom  # [sample]
            sk = (t2 - t_k) * (t_k - t1) / (t2 - t1 + 1e-8)  # [sample]

            std_x = torch.sqrt(torch.clamp(x_var * sk * dt, min=0.0))  # [sample]
            noise_x = torch.randn(D, sample, device=device) * std_x  # [3, sample]
            Xs[:, :, k] = ak.unsqueeze(0) * Xs[:, :, k - 1] + b_val * r.unsqueeze(0) + noise_x

            noise_y = torch.randn(D, sample, device=device) * sqrt_y_dt
            Ys[:, :, k] = Ys[:, :, k - 1] + Xs[:, :, k - 1] * dt * scale_adj + noise_y

            # ---------------- 3) Weights & resampling (vectorized) ----------------
            # z_pred: [N, sample]
            z_pred = W @ Ys[:, :, k]  # [N, sample]
            diff = Z[k - 1].unsqueeze(1) - z_pred  # [N, sample]
            exponent = -0.5 * (diff * diff * inv_diag.unsqueeze(1)).sum(dim=0)  # [sample]
            weights = (norm_const * torch.exp(exponent)).clamp_min(tiny)  # [sample]
            ws = weights / (weights.sum() + tiny)

            idxs = torch.multinomial(ws, sample, replacement=True)

            Ts = Ts[idxs]
            Cntr = Cntr[idxs]
            Xs = Xs[:, idxs, :]
            Ys = Ys[:, idxs, :]
            Ms = Ms[:, idxs, :]

        return {'Xs': Xs, 'Ys': Ys, 'Ms': Ms, 'Ts': Ts, 'Cntr': Cntr}
    @staticmethod
    def map_model_fit(x, a0, b0, c0, d0):
        """
        MAP estimation of Gamma(shape, scale) parameters given data x and Gamma priors:
            alpha ~ Gamma(a0, b0)
            theta ~ Gamma(c0, d0)
        Returns: alpha_map, theta_map
        """
        # Convert to numpy
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().ravel()
        else:
            x_np = np.asarray(x).ravel()

        N = x_np.size
        S = np.sum(x_np)
        L = np.sum(np.log(x_np))

        def neg_log_post(params):
            alpha, theta = params
            if alpha <= 0 or theta <= 0:
                return np.inf
            # use math.lgamma for log Γ(alpha)
            lp = 0.0
            lp -= N * math.lgamma(alpha)
            lp -= N * alpha * math.log(theta)
            lp += (alpha - 1) * L
            lp -= S / theta
            lp += (a0 - 1) * math.log(alpha)
            lp -= alpha / b0
            lp += (c0 - 1) * math.log(theta)
            lp -= theta / d0
            return -lp

        # Initial guess
        init = np.array([1.0, float(np.mean(x_np))])
        bounds = [(1e-5, None), (1e-5, None)]
        res = minimize(
            neg_log_post,
            init,
            method='L-BFGS-B',
            bounds=bounds
        )
        return float(res.x[0]), float(res.x[1])

    def em_step(self, Z: torch.Tensor, data: Dict[str, torch.Tensor]) -> None:
        """
        Perform one EM update of {x_mu0, x_var0, y_mu0, y_var0,
        mark_mean, mark_var, gam_alpha, gam_beta, event_cnt}
        in-place on self.
        Args:
          Z    : Tensor[K, N]    (not used in M-step here)
          data : dict with keys 'Xs','Ys','Ms','Ts','Cntr'
        """
        device = self.device
        sample = self.sample

        # Unpack
        Xs = data['Xs']  # [3, sample, K+1]
        Ys = data['Ys']  # [3, sample, K+1]
        Ms = data['Ms']  # [3, sample, max_events]
        Ts = data['Ts']  # [sample, max_events]
        Cntr = data['Cntr']  # [sample]

        # Hyperpriors
        norm_mu0, norm_k0 = self.norm_mu0, self.norm_k0
        norm_phi, norm_v0 = self.norm_phi, self.norm_v0
        a0, b0, c0, d0 = self.gam_a0, self.gam_b0, self.gam_c0, self.gam_d0

        # 1) Update X(0) prior
        xt0 = Xs[:, :, 0]  # [3, sample]
        mx = xt0.mean(dim=1)  # [3]
        self.x_mu0 = (norm_k0 * norm_mu0 + sample * mx) / (norm_k0 + sample)

        diff = xt0 - mx.unsqueeze(1)  # [3, sample]
        S = diff @ diff.t()  # [3,3]
        P = (mx - norm_mu0).unsqueeze(1) @ (mx - norm_mu0).unsqueeze(0)
        tmp = norm_phi + S + P * (norm_k0 * sample) / (norm_k0 + sample)
        self.x_var0 = tmp / (norm_v0 + sample + 3 + 2)

        # 2) Update Y(0) prior (same form)
        yt0 = Ys[:, :, 0]
        my = yt0.mean(dim=1)
        self.y_mu0 = (norm_k0 * norm_mu0 + sample * my) / (norm_k0 + sample)

        diff = yt0 - my.unsqueeze(1)
        S = diff @ diff.t()
        P = (my - norm_mu0).unsqueeze(1) @ (my - norm_mu0).unsqueeze(0)
        tmp = norm_phi + S + P * (norm_k0 * sample) / (norm_k0 + sample)
        self.y_var0 = tmp / (norm_v0 + sample + 3 + 2)

        # 3) Per-event mark & waiting‐time updates
        max_cntr = int(Cntr.max().item())
        for e in range(2, max_cntr + 1):
            inds = torch.where(Cntr >= e)[0]  # shape (t_count,)
            t_count = inds.numel()

            if t_count > 0:
                # Ms has shape [3, sample, max_events]
                # Ms[:, inds, e-1] will now be [3, t_count], even if t_count==1
                m_e = Ms[:, inds, e - 1]  # [3, t_count]
                m_mean = m_e.mean(dim=1)  # [3]

                # standard NIW update
                self.mark_mean[e - 2] = (norm_k0 * norm_mu0 + t_count * m_mean) / (norm_k0 + t_count)
                diff = m_e - m_mean.unsqueeze(1)
                S = diff @ diff.t()
                P = (m_mean - norm_mu0).unsqueeze(1) @ (m_mean - norm_mu0).unsqueeze(0)
                tmp = norm_phi + S + P * (norm_k0 * t_count) / (norm_k0 + t_count)
                self.mark_var[e - 2] = tmp / (norm_v0 + t_count + 3 + 2)

                # waiting‐time update unchanged
                ts = Ts[inds, e - 1] - Ts[inds, e - 2]
                alpha_map, theta_map = self.map_model_fit(ts, a0, b0, c0, d0)
                self.gam_alpha[e - 2] = alpha_map
                self.gam_beta[e - 2] = theta_map

        # 4) Update event count
        self.event_cnt = max_cntr - 1

    def generate(self, cur_param) -> Dict[str, torch.Tensor]:
        """
        Simulate multivariate (3D) SSM trajectories using learned HiSDE parameters.
        Returns dict with:
          Xs   : [3 × sample × (K+1)] latent increments
          Ys   : [3 × sample × (K+1)] integrator trajectories
          Ms   : [3 × sample × max_events] event marks
          Ts   : [sample × max_events] event times
          Cntr : [sample]           count of events per trajectory
        """
        device = self.device
        D = 3
        sample = self.sample
        K = self.K
        dt = self.dt
        x_var = self.x_var
        y_var = self.y_var
        scale_adj = self.scale_adj

        # Gamma‐prior for waiting times
        gam_alpha_prior = self.gam_a0 / self.gam_b0
        gam_beta_prior = self.gam_c0 / self.gam_d0

        # NIW priors for marks before any events
        norm_mu0, norm_phi = self.norm_mu0, self.norm_phi

        # Pre‐allocate outputs
        Xs = torch.zeros(D, sample, K + 1, device=device)
        Ys = torch.zeros(D, sample, K + 1, device=device)
        Ms = torch.zeros(D, sample, self.max_events, device=device)
        Ts = torch.zeros(sample, self.max_events, device=device)
        Cntr = torch.ones(sample, dtype=torch.long, device=device)

        # --- Initialize at k = 0 ---
        mvn_x0 = torch.distributions.MultivariateNormal(
            cur_param.x_mu0, covariance_matrix=cur_param.x_var0
        )
        mvn_y0 = torch.distributions.MultivariateNormal(
            cur_param.y_mu0, covariance_matrix=cur_param.y_var0
        )
        x0 = mvn_x0.sample((sample,))  # [sample × 3]
        y0 = mvn_y0.sample((sample,))
        Xs[:, :, 0] = x0.T
        Ys[:, :, 0] = y0.T
        # Ms[:,:,0] and Ts[:,0] are already zero

        # --- Simulate forward ---
        for k in range(1, K + 1):
            t_k = k * dt
            for s in range(sample):
                cnt = Cntr[s].item()
                last_t = Ts[s, cnt - 1]

                # 1) Propose a new event if we've passed the last one
                if last_t < t_k:
                    if cnt <= cur_param.event_cnt:
                        a = cur_param.gam_alpha[cnt - 1]
                        b = cur_param.gam_beta[cnt - 1]
                    else:
                        a, b = gam_alpha_prior, gam_beta_prior
                    gamma_dist = torch.distributions.Gamma(concentration=a, rate=1.0 / b)
                    wait = gamma_dist.sample()
                    wait = torch.clamp(wait, min=2 * dt)
                    Ts[s, cnt] = last_t + wait

                    # Sample a 3D mark
                    if cnt <= cur_param.event_cnt:
                        mu_m = cur_param.mark_mean[cnt - 1]  # [3]
                        cov_m = cur_param.mark_var[cnt - 1]  # [3×3]
                    else:
                        mu_m, cov_m = norm_mu0, norm_phi
                    mvn_m = torch.distributions.MultivariateNormal(mu_m, covariance_matrix=cov_m)
                    Ms[:, s, cnt] = mvn_m.sample()

                    Cntr[s] = cnt + 1

                # 2) Propagate latent increment X and integrator Y
                idx = Cntr[s].item() - 1
                t2 = Ts[s, idx]
                t1 = Ts[s, idx - 1] if idx > 0 else torch.tensor(0.0, device=device)

                denom = torch.clamp(t2 - t_k, min=0.5 * dt)
                ak = 1.0 - dt / denom
                bk = Ms[:, s, idx] * (dt / denom)
                sk = (t2 - t_k) * (t_k - t1) / (t2 - t1 + 1e-8)

                noise_x = torch.randn(D, device=device) * math.sqrt(x_var * sk * dt)
                Xs[:, s, k] = ak * Xs[:, s, k - 1] + bk + noise_x

                noise_y = torch.randn(D, device=device) * math.sqrt(y_var * dt)
                Ys[:, s, k] = (
                        Ys[:, s, k - 1]
                        + Xs[:, s, k - 1] * dt * scale_adj
                        + noise_y
                )

        return {'Xs': Xs, 'Ys': Ys, 'Ms': Ms, 'Ts': Ts, 'Cntr': Cntr}

    def model_z_3d(self,
                   y0: Tuple[float, float, float] = (-1.0, 1.0, -1.0),
                   lorenz_dt: float = 0.01
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate 3D Lorenz latent trajectories and noisy observations.

        Returns:
            Z: Tensor of shape (N, K)   – noisy observations
            Y: Tensor of shape (3, K)   – true centered latent states
        """
        # 1) simulate true Lorenz at fine dt
        t_span = (0.0, (self.K - 1) * lorenz_dt)
        t_uniform, y_uniform = simulate_lorenz(t_span, lorenz_dt, y0)
        # y_uniform: np.ndarray of shape (K,3)

        # 2) convert & center
        Y = torch.from_numpy(y_uniform.T).float().to(self.device)  # [3, K]
        Y = Y - Y.mean(dim=1, keepdim=True)

        # 3) project to observations and add noise
        #   noise ~ N(0, S) where S = self.z_var
        L = torch.linalg.cholesky(self.z_var)  # [N×N]
        noise = L @ torch.randn(self.N, self.K, device=self.device)

        Z = self.W @ Y + noise  # [N×K]

        return Z, Y


def plot_hisde_3d(Z, Y_fit, Y_true, Xs, Ms, dt, tau_idx):
    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)

    Z_np = to_np(Z)
    Y_true_np = to_np(Y_true)
    Yf = to_np(Y_fit)
    if Yf.ndim == 3 and Yf.shape[0] != 3:
        Yf = Yf.transpose(1, 0, 2)
    Y_fit_mean = Yf.mean(axis=1)

    M_all = to_np(Ms)[0, 0, :]
    n_events = len(tau_idx)
    marks = M_all[:n_events]

    # —— 1) Heatmap as Plotly ——
    fig1 = go.Figure(go.Heatmap(
        z=Z_np,
        colorbar=dict(title="Intensity")
    ))
    fig1.update_layout(
        title="Observed data heatmap",
        xaxis_title="Time index",
        yaxis_title="Feature / Dimension",
        margin=dict(l=40, r=20, t=40, b=40)
    )

    # —— 2) Interactive 3D (your existing fig2) ——
    fig2 = go.Figure()
    P = Y_true_np[:, tau_idx]
    # inferred mean
    fig2.add_trace(go.Scatter3d(
        x=Y_fit_mean[0], y=Y_fit_mean[1], z=Y_fit_mean[2],
        mode='lines', name='Inferred mean'
    ))
    # true
    fig2.add_trace(go.Scatter3d(
        x=Y_true_np[0], y=Y_true_np[1], z=Y_true_np[2],
        mode='lines', line=dict(dash='dash'), name='True'
    ))
    # events
    fig2.add_trace(go.Scatter3d(
        x=P[0], y=P[1], z=P[2],
        mode='markers', marker=dict(size=4, color='red'),
        name='Events'
    ))
    # optional line‐arrows
    for i in range(P.shape[1] - 1):
        fig2.add_trace(go.Scatter3d(
            x=[P[0, i], P[0, i + 1]],
            y=[P[1, i], P[1, i + 1]],
            z=[P[2, i], P[2, i + 1]],
            mode='lines', line=dict(width=2, color='black'),
            showlegend=False
        ))
    fig2.update_layout(
        scene=dict(
            xaxis_title='Y₁', yaxis_title='Y₂', zaxis_title='Y₃'
        ),
        title='Interactive 3D latent trajectories with inducing points',
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # —— 3) Marks‐vs‐τ as Plotly ——
    fig3 = go.Figure(go.Scatter(
        x=tau_idx, y=marks, mode='markers+lines'
    ))
    fig3.update_layout(
        title="Mark vs. τ",
        xaxis_title="Event time index (τ)",
        yaxis_title="Mark value (dim 1)",
        margin=dict(l=40, r=20, t=40, b=40)
    )

    return fig1, fig2, fig3


device = 'cuda'

# 1) Initialize 3D HiSDE model
model = HiSDE3D(device=device)

# 2) Generate synthetic 3D data
Z, Y_true = model.model_z_3d()  # Z: [N,K], Y_true: [3,K]



# 3) Inference loop: alternate SMC and EM
in_loop = 2  # number of SMC+EM updates per iteration
num_iters = 10  # outer EM iterations

in_start = 1  # rolling-window counter, exactly like MATLAB

for it in range(1, num_iters + 1):
    for l in range(in_loop):
        # growing window size
        window_end = min(100 + it * 100, model.K)
        # smc expects shape [T, N]
        Z_window = Z[:, :window_end].T

        # 3a) Sequential Monte Carlo
        #data = model.smc(iteration=in_start, Z=Z_window, cur_param=model)
        data = model.smc_fast(iteration=in_start, Z=Z_window, cur_param=model)
        # 3b) EM update (in-place on model)
        model.em_step(Z=Z_window, data=data)

        # increment the rolling‐window counter
        in_start += 1

        print(f"  inner loop {l + 1}/{in_loop} (SMC iter={in_start - 1})")

# 4) Extract final trajectories & marks
Xs = data['Xs']  # [3, sample, K+1]
Ys = data['Ys']  # [3, sample, K+1]
Ms = data['Ms']  # [3, sample, max_events]   <— NEW

# 5) Compute inducing‐point indices for plotting
raw_tau = data['Ts'][0, : model.event_cnt]
tau_idx = (raw_tau / model.dt).round().long().cpu().numpy().astype(int)

# 6) Drop t=0 and convert to numpy
T = Z.shape[1]
Ys_plot = Ys[:, :, 1:T + 1]
Xs_plot = Xs[:, :, 1:T + 1]

Z_np = Z.cpu().numpy()
Ys_np = Ys_plot.cpu().numpy()
Y_true_np = Y_true.cpu().numpy()
Xs_np = Xs_plot.cpu().numpy()
Ms_np = Ms.cpu().numpy()  # <— NEW

# 7) Call the plotting function with Ms included
fig1, fig2, fig3 = plot_hisde_3d(
    Z_np,
    Ys_np,
    Y_true_np,
    Xs_np,
    Ms_np,
    model.dt,
    tau_idx
)
for f in (fig1, fig2, fig3):
    f.show()




def plot_hisde_3d_matplotlib(Z, Y_fit, Y_true, Xs, Ms, dt, tau_idx):
    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)

    # Convert tensors to numpy
    Z_np = to_np(Z)
    Y_true_np = to_np(Y_true)
    Yf = to_np(Y_fit)
    if Yf.ndim == 3 and Yf.shape[0] != 3:
        Yf = Yf.transpose(1, 0, 2)
    Y_fit_mean = Yf.mean(axis=1)

    M_all = to_np(Ms)[0, 0, :]
    n_events = len(tau_idx)
    marks = M_all[:n_events]

    figs = []

    # --- 1) Heatmap ---
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    im = ax1.imshow(Z_np, aspect='auto', origin='lower', cmap='viridis')
    cbar = fig1.colorbar(im, ax=ax1)
    cbar.set_label("Intensity")
    ax1.set_title("Observed data heatmap")
    ax1.set_xlabel("Time index")
    ax1.set_ylabel("Feature / Dimension")
    figs.append(fig1)

    # --- 2) 3D latent trajectories ---
    fig2 = plt.figure(figsize=(6, 5))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Inferred mean
    ax2.plot(Y_fit_mean[0], Y_fit_mean[1], Y_fit_mean[2], label="Inferred mean")

    # True
    ax2.plot(Y_true_np[0], Y_true_np[1], Y_true_np[2], linestyle='--', label="True")

    # Events
    P = Y_true_np[:, tau_idx]
    ax2.scatter(P[0], P[1], P[2], c='red', s=20, label="Events")

    # Optional connecting lines between events
    for i in range(P.shape[1] - 1):
        ax2.plot(
            [P[0, i], P[0, i + 1]],
            [P[1, i], P[1, i + 1]],
            [P[2, i], P[2, i + 1]],
            color='black', linewidth=1
        )

    ax2.set_xlabel('Y₁')
    ax2.set_ylabel('Y₂')
    ax2.set_zlabel('Y₃')
    ax2.set_title('3D latent trajectories with inducing points')
    ax2.legend()
    figs.append(fig2)

    # --- 3) Mark vs τ ---
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(tau_idx, marks, marker='o', linestyle='-')
    ax3.set_title("Mark vs. τ")
    ax3.set_xlabel("Event time index (τ)")
    ax3.set_ylabel("Mark value (dim 1)")
    figs.append(fig3)

    return figs
figs = plot_hisde_3d_matplotlib(Z, Y_fit, Y_true, Xs, Ms, dt, tau_idx)
plt.show()  # This will show all 3 in PyCharm’s SciView