import math
from typing import Dict, Tuple

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# --------------- small torch helpers (no scipy/sklearn) ---------------

def torch_chirp_linear(t: torch.Tensor, f0: float, t1: float, f1: float) -> torch.Tensor:
    """
    Linear chirp like scipy.signal.chirp(..., method='linear'), returning cos(phase).
    phase(t) = 2*pi*( f0*t + 0.5*((f1-f0)/t1)*t^2 )
    """
    k = (f1 - f0) / t1
    phase = 2.0 * math.pi * (f0 * t + 0.5 * k * t * t)
    return torch.cos(phase)


def rk4(f, y0: torch.Tensor, t0: float, t1: float, dt: float, device=None) -> torch.Tensor:
    """
    Simple RK4 integrator in torch. Returns states for each step including the first one.
    y0: (...,) tensor; f(t, y) -> (...,) tensor
    """
    if device is None:
        device = y0.device
    n_steps = int(math.floor((t1 - t0) / dt))
    ys = [y0]
    t = t0
    y = y0
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = f(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        ys.append(y)
    return torch.stack(ys, dim=0)  # (n_steps+1, ...)


# -------------------- tiny GP in torch (RBF + Linear + Bias) --------------------

class TorchGP:
    """
    Minimal GP:
      k(x,x') = σ_rbf^2 * exp(-0.5 * ||x-x'||^2 / ℓ^2) + σ_lin^2 * (x x') + σ_bias^2
    Trains (ℓ, σ_rbf, σ_lin, σ_bias) by maximizing log marginal likelihood (LBFGS).
    Noise variance is fixed to the provided alpha (like sklearn's 'alpha').
    """

    def __init__(self, alpha: float = 1e-4, device='cpu'):
        self.device = torch.device(device)
        self.log_ell = torch.nn.Parameter(torch.tensor(2.0, device=self.device))
        self.log_s_rbf = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_s_lin = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        self.log_s_bias = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        self.alpha = torch.tensor(alpha, device=self.device)
        self.x_train = None
        self.y_train = None
        self.L = None  # Cholesky of K + α I

    @staticmethod
    def _cdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = (x * x).sum(dim=-1, keepdim=True)  # (n,1)
        y2 = (y * y).sum(dim=-1, keepdim=True).T  # (1,m)
        return x2 + y2 - 2.0 * (x @ y.T)

    def kernel(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        ell = torch.exp(self.log_ell) + 1e-12
        s_rbf = torch.exp(self.log_s_rbf)
        s_lin = torch.exp(self.log_s_lin)
        s_bias = torch.exp(self.log_s_bias)
        d2 = self._cdist2(xa, xb) / (ell * ell)
        rbf = (s_rbf * s_rbf) * torch.exp(-0.5 * d2)
        lin = (s_lin * s_lin) * (xa @ xb.T)
        bias = (s_bias * s_bias)
        return rbf + lin + bias

    def _mll(self, K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # K is (n,n) = k(X,X) + α I
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y, L)  # K^{-1} y
        n = y.shape[0]
        mll = -0.5 * (y.T @ alpha) - torch.sum(torch.log(torch.diagonal(L))) - 0.5 * n * math.log(2 * math.pi)
        return mll.squeeze(), L

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor, steps: int = 80) -> "TorchGP":
        x_train = x_train.to(self.device).float()
        y_train = y_train.to(self.device).float().view(-1, 1)
        self.x_train, self.y_train = x_train, y_train

        opt = torch.optim.LBFGS([self.log_ell, self.log_s_rbf, self.log_s_lin, self.log_s_bias],
                                lr=0.5, max_iter=steps, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            K = self.kernel(x_train, x_train)
            K = K + self.alpha * torch.eye(K.shape[0], device=self.device)
            # Add tiny jitter for numerical stability
            K = K + 1e-6 * torch.eye(K.shape[0], device=self.device)
            mll, L = self._mll(K, y_train)
            loss = -mll
            loss.backward()
            return loss

        opt.step(closure)
        # Cache final L
        with torch.no_grad():
            K = self.kernel(x_train, x_train)
            K = K + (self.alpha + 1e-6) * torch.eye(K.shape[0], device=self.device)
            self.L = torch.linalg.cholesky(K)
        return self

    def predict(self, x_test: torch.Tensor, return_std: bool = True):
        x_test = x_test.to(self.device).float()
        Kx = self.kernel(self.x_train, x_test)  # (n, m)
        # Solve for K^{-1} y  and  K^{-1} Kx
        Kinv_y = torch.cholesky_solve(self.y_train, self.L)  # (n,1)
        Kinv_Kx = torch.cholesky_solve(Kx, self.L)  # (n,m)

        mean = (Kx.T @ Kinv_y).squeeze(-1)  # (m,)

        if return_std:
            Kxx = self.kernel(x_test, x_test)
            cov = Kxx - (Kx.T @ Kinv_Kx)
            # Numerical floor to avoid tiny negative variances
            var = torch.clamp(torch.diagonal(cov), min=1e-12)
            std = torch.sqrt(var)
            return mean, std
        else:
            return mean, None


# -------------------------- HiSDE1D (torch-only) --------------------------

class HiSDE1D:
    """
    Hierarchical SDE model in 1D — torch-only version.
    """

    def __init__(
            self,
            max_events: int = 100,
            gam_a0: float = 20.0,
            gam_b0: float = 2.0,
            gam_c0: float = 5.0,
            gam_d0: float = 6.0,
            norm_mu0: float = 0.0,
            norm_k0: float = 1.0,
            norm_alpha: float = 3.0,
            norm_beta: float = 100.0,
            K: int = 500,
            dt: float = 0.5,
            sample: int = 10000,
            x_var: float = 1e-1,
            y_var: float = 1e-4,
            z_var: float = 0.1,
            scale_adj: float = 1.0,
            device: str = 'cpu'
    ):
        self.max_events = max_events
        self.gam_a0 = gam_a0
        self.gam_b0 = gam_b0
        self.gam_c0 = gam_c0
        self.gam_d0 = gam_d0
        self.norm_mu0 = norm_mu0
        self.norm_k0 = norm_k0
        self.norm_alpha = norm_alpha
        self.norm_beta = norm_beta
        self.K = K
        self.dt = dt
        self.sample = sample
        self.x_var = x_var
        self.y_var = y_var
        self.z_var = z_var
        self.scale_adj = scale_adj

        self.device = torch.device(device)

        # EM-updated parameters
        self.mark_mean = torch.zeros(self.max_events, device=self.device)
        self.mark_var = torch.zeros(self.max_events, device=self.device)
        self.gam_alpha = torch.zeros(self.max_events, device=self.device)
        self.gam_beta = torch.zeros(self.max_events, device=self.device)
        self.event_cnt = 0

    # ---------------- Simulation ----------------

    def generate(self, cur_param) -> Dict[str, torch.Tensor]:
        device = self.device

        scale_adj = self.scale_adj
        max_events = self.max_events
        gam_alpha_prior = self.gam_a0 / self.gam_b0
        gam_beta_prior = self.gam_c0 / self.gam_d0
        norm_mu0 = self.norm_mu0
        norm_k0 = self.norm_k0
        norm_alpha = self.norm_alpha
        norm_beta = self.norm_beta
        K = self.K
        dt = self.dt
        sample = self.sample
        x_var = self.x_var
        y_var = self.y_var

        Xs = torch.zeros(sample, K + 1, device=device)
        Ys = torch.zeros(sample, K + 1, device=device)
        Ms = torch.zeros(sample, max_events, device=device)
        Ts = torch.zeros(sample, max_events, device=device)
        Cntr = torch.ones(sample, dtype=torch.long, device=device)

        # init
        Xs[:, 0] = cur_param.x_mu0 + torch.randn(sample, device=device) * math.sqrt(cur_param.x_var0)
        Ys[:, 0] = cur_param.y_mu0 + torch.randn(sample, device=device) * math.sqrt(cur_param.y_var0)

        gamma_prior = torch.distributions.Gamma(concentration=gam_alpha_prior, rate=1.0 / gam_beta_prior)

        for k in range(1, K + 1):
            t_k = k * dt
            for s in range(sample):
                cur_cntr = int(Cntr[s].item())
                if Ts[s, cur_cntr - 1] < t_k:
                    if cur_cntr <= cur_param.event_cnt:
                        a = cur_param.gam_alpha[cur_cntr - 1]
                        b = cur_param.gam_beta[cur_cntr - 1]
                        gamma_dist = torch.distributions.Gamma(concentration=a, rate=1.0 / b)
                    else:
                        gamma_dist = gamma_prior

                    next_wait = gamma_dist.sample()
                    next_wait = torch.maximum(torch.tensor(2 * dt, device=device), next_wait)
                    Ts[s, cur_cntr] = torch.floor(Ts[s, cur_cntr - 1] + next_wait) + 0.5 * dt

                    if cur_cntr <= cur_param.event_cnt:
                        Ms[s, cur_cntr] = cur_param.mark_mean[cur_cntr - 1] + torch.randn((),
                                                                                          device=device) * math.sqrt(
                            cur_param.mark_var[cur_cntr - 1])
                    else:
                        mark_prior_std = math.sqrt(norm_beta / ((norm_alpha - 1) * norm_k0))
                        Ms[s, cur_cntr] = norm_mu0 + torch.randn((), device=device) * mark_prior_std

                    Cntr[s] = cur_cntr + 1

                t2 = Ts[s, Cntr[s] - 1]
                t1 = Ts[s, Cntr[s] - 2] if Cntr[s] >= 2 else torch.tensor(0.0, device=device)
                b_val = Ms[s, Cntr[s] - 1]

                denom = torch.maximum(torch.tensor(0.5 * dt, device=device), t2 - t_k)
                ak = 1 - dt / denom
                bk = b_val * dt / denom
                sk = (t2 - t_k) * (t_k - t1) / (t2 - t1 + 1e-8)

                Xs[s, k] = ak * Xs[s, k - 1] + bk + math.sqrt(x_var * sk * dt) * torch.randn((), device=device)
                Ys[s, k] = Ys[s, k - 1] + Xs[s, k - 1] * dt * scale_adj + math.sqrt(y_var * dt) * torch.randn((),
                                                                                                              device=device)

        return {'Xs': Xs, 'Ys': Ys, 'Ms': Ms, 'Ts': Ts, 'Cntr': Cntr}

    # ---------------- SMC ----------------

    def smc(self, iter: int, Z: torch.Tensor, cur_param) -> Dict[str, torch.Tensor]:
        device = self.device
        scale_adj = self.scale_adj
        max_events = self.max_events
        dt = self.dt
        sample = self.sample
        x_var, y_var = self.x_var, self.y_var
        z_var = self.z_var

        gam_alpha_prior = self.gam_a0 / self.gam_b0
        gam_beta_prior = self.gam_c0 / self.gam_d0
        norm_mu0, norm_k0 = self.norm_mu0, self.norm_k0
        norm_alpha, norm_beta = self.norm_alpha, self.norm_beta

        K = Z.numel()

        Xs = torch.zeros(sample, K + 1, device=device)
        Ys = torch.zeros(sample, K + 1, device=device)
        Ms = torch.zeros(sample, max_events, device=device)
        Ts = torch.zeros(sample, max_events, device=device)
        Cntr = torch.ones(sample, dtype=torch.long, device=device)

        if iter == 1:
            prior_std = math.sqrt(norm_beta / ((norm_alpha - 1) * norm_k0))
            Xs[:, 0] = norm_mu0 + torch.randn(sample, device=device) * prior_std
            Ys[:, 0] = norm_mu0 + torch.randn(sample, device=device) * prior_std
        else:
            Xs[:, 0] = cur_param.x_mu0 + torch.randn(sample, device=device) * math.sqrt(cur_param.x_var0)
            Ys[:, 0] = cur_param.y_mu0 + torch.randn(sample, device=device) * math.sqrt(cur_param.y_var0)

        for k in tqdm(range(1, K + 1), desc="Sequential update over k = 0...K-1"):
            t_k = k * dt
            weights = torch.zeros(sample, device=device)

            for s in range(sample):
                cnt = int(Cntr[s].item())

                if Ts[s, cnt - 1] < t_k:
                    if cnt <= cur_param.event_cnt:
                        a = cur_param.gam_alpha[cnt - 1]
                        b = cur_param.gam_beta[cnt - 1]
                    else:
                        a, b = gam_alpha_prior, gam_beta_prior
                    gamma_dist = torch.distributions.Gamma(concentration=a, rate=1.0 / b)
                    wait = torch.maximum(gamma_dist.sample(), torch.tensor(2 * dt, device=device))

                    Ts[s, cnt] = Ts[s, cnt - 1] + wait
                    if cnt <= cur_param.event_cnt:
                        Ms[s, cnt] = cur_param.mark_mean[cnt - 1] + torch.randn((), device=device) * math.sqrt(
                            cur_param.mark_var[cnt - 1])
                    else:
                        prior_mark_std = math.sqrt(norm_beta / ((norm_alpha - 1) * norm_k0))
                        Ms[s, cnt] = self.norm_mu0 + torch.randn((), device=device) * prior_mark_std
                    Cntr[s] = cnt + 1

                t2 = Ts[s, Cntr[s] - 1]
                t1 = Ts[s, Cntr[s] - 2] if Cntr[s] > 1 else torch.tensor(0.0, device=device)
                b_val = Ms[s, Cntr[s] - 1]

                denom = torch.maximum(torch.tensor(0.5 * dt, device=device), t2 - t_k)
                ak = 1 - dt / denom
                bk = b_val * dt / denom
                sk = (t2 - t_k) * (t_k - t1) / (t2 - t1 + 1e-8)

                Xs[s, k] = ak * Xs[s, k - 1] + bk + math.sqrt(x_var * sk * dt) * torch.randn((), device=device)
                Ys[s, k] = Ys[s, k - 1] + Xs[s, k - 1] * dt * scale_adj + math.sqrt(y_var * dt) * torch.randn((),
                                                                                                              device=device)

                diff = Z[k - 1] - Ys[s, k]
                weights[s] = torch.exp(-0.5 * diff * diff / z_var) / math.sqrt(2 * math.pi * z_var)

            ws = weights / (weights.sum() + 1e-12)
            idx = torch.multinomial(ws, sample, replacement=True)

            Xs, Ys = Xs[idx], Ys[idx]
            Ms, Ts = Ms[idx], Ts[idx]
            Cntr = Cntr[idx]

        return {'Xs': Xs, 'Ys': Ys, 'Ms': Ms, 'Ts': Ts, 'Cntr': Cntr}

    def smc_fast(self, iter: int, Z: torch.Tensor, cur_param) -> Dict[str, torch.Tensor]:
        device = self.device
        scale_adj = self.scale_adj
        max_events = self.max_events
        dt = self.dt
        sample = self.sample
        x_var, y_var = self.x_var, self.y_var
        z_var = self.z_var

        gam_alpha_prior = self.gam_a0 / self.gam_b0
        gam_beta_prior = self.gam_c0 / self.gam_d0
        norm_mu0, norm_k0 = self.norm_mu0, self.norm_k0
        norm_alpha, norm_beta = self.norm_alpha, self.norm_beta

        K = Z.numel()

        # Allocate
        Xs = torch.zeros(sample, K + 1, device=device)
        Ys = torch.zeros(sample, K + 1, device=device)
        Ms = torch.zeros(sample, max_events, device=device)
        Ts = torch.zeros(sample, max_events, device=device)
        Cntr = torch.ones(sample, dtype=torch.long, device=device)

        # Init X(0), Y(0)
        if iter == 1:
            prior_std = math.sqrt(norm_beta / ((norm_alpha - 1) * norm_k0))
            Xs[:, 0] = norm_mu0 + torch.randn(sample, device=device) * prior_std
            Ys[:, 0] = norm_mu0 + torch.randn(sample, device=device) * prior_std
        else:
            Xs[:, 0] = cur_param.x_mu0 + torch.randn(sample, device=device) * math.sqrt(cur_param.x_var0)
            Ys[:, 0] = cur_param.y_mu0 + torch.randn(sample, device=device) * math.sqrt(cur_param.y_var0)

        rows = torch.arange(sample, device=device)
        half_dt = 0.5 * dt
        sqrt_y_dt = math.sqrt(y_var * dt)
        tiny = 1e-8
        two_dt = 2.0 * dt
        prior_mark_std = math.sqrt(norm_beta / ((norm_alpha - 1) * norm_k0))

        for k in range(1, K + 1):
            t_k = k * dt

            # --------- 1) Event updates (vectorized over particles) ----------
            # Is the next scheduled time earlier than current time?
            last_idx = Cntr - 1
            t_last = Ts[rows, last_idx]
            event_mask = t_last < t_k

            if event_mask.any():
                # Choose per-particle Gamma params (known vs prior)
                if cur_param.event_cnt > 0:
                    known_mask = (Cntr <= cur_param.event_cnt) & event_mask
                else:
                    known_mask = torch.zeros_like(event_mask)

                # default to priors
                a = torch.full((sample,), gam_alpha_prior, device=device)
                b = torch.full((sample,), gam_beta_prior, device=device)

                # fill known with learned params (gather by event index)
                if known_mask.any():
                    idx_known_ev = (Cntr[known_mask] - 1)
                    a[known_mask] = cur_param.gam_alpha[idx_known_ev]
                    b[known_mask] = cur_param.gam_beta[idx_known_ev]

                # sample waits only for those with an event
                a_ev = a[event_mask]
                b_ev = b[event_mask]
                # Gamma with 'rate' = 1/scale
                gamma_ev = torch.distributions.Gamma(concentration=a_ev, rate=1.0 / b_ev).sample()
                wait_ev = torch.maximum(gamma_ev, torch.full_like(gamma_ev, two_dt))

                rows_ev = rows[event_mask]
                cols_ev = Cntr[event_mask]

                # update Ts (next event time) at current counter
                Ts[rows_ev, cols_ev] = Ts[rows_ev, cols_ev - 1] + wait_ev

                # sample/update marks at current counter
                ms_ev = torch.empty_like(wait_ev)
                if cur_param.event_cnt > 0:
                    known_ev = (Cntr[event_mask] <= cur_param.event_cnt)
                else:
                    known_ev = torch.zeros_like(event_mask[event_mask])

                if known_ev.any():
                    idxm = (Cntr[event_mask][known_ev] - 1)
                    ms_ev[known_ev] = (
                            cur_param.mark_mean[idxm]
                            + torch.randn(known_ev.sum(), device=device)
                            * torch.sqrt(cur_param.mark_var[idxm].clamp_min(0.0))
                    )
                if (~known_ev).any():
                    ms_ev[~known_ev] = (
                            self.norm_mu0
                            + torch.randn((~known_ev).sum(), device=device) * prior_mark_std
                    )

                Ms[rows_ev, cols_ev] = ms_ev
                Cntr[event_mask] = Cntr[event_mask] + 1

            # --------- 2) Continuous state update (vectorized) ----------
            # Gather t2 (last event time), t1 (prev event time or 0), and b (last mark)
            last_idx = Cntr - 1
            t2 = Ts[rows, last_idx]

            t1 = torch.zeros(sample, device=device)
            has_prev = Cntr > 1
            prev_idx = Cntr[has_prev] - 2
            t1[has_prev] = Ts[rows[has_prev], prev_idx]

            b_val = Ms[rows, last_idx]

            denom = torch.maximum(torch.full((sample,), half_dt, device=device), t2 - t_k)
            ak = 1.0 - dt / denom
            bk = b_val * dt / denom
            sk = (t2 - t_k) * (t_k - t1) / (t2 - t1 + tiny)

            # Sample process noise (vectorized)
            eps_x = torch.randn(sample, device=device)
            eps_y = torch.randn(sample, device=device)

            Xs[:, k] = ak * Xs[:, k - 1] + bk + torch.sqrt(torch.clamp(x_var * sk * dt, min=0.0)) * eps_x
            Ys[:, k] = Ys[:, k - 1] + Xs[:, k - 1] * dt * scale_adj + sqrt_y_dt * eps_y

            # --------- 3) Weights & resampling (vectorized) ----------
            diff = Z[k - 1] - Ys[:, k]
            # unnormalized Gaussian likelihood
            weights = torch.exp(-0.5 * diff * diff / z_var) / math.sqrt(2.0 * math.pi * z_var)
            ws = weights / (weights.sum() + tiny)

            idx = torch.multinomial(ws, sample, replacement=True)

            # Resample EVERYTHING to keep genealogy identical to original semantics
            Xs = Xs[idx]
            Ys = Ys[idx]
            Ms = Ms[idx]
            Ts = Ts[idx]
            Cntr = Cntr[idx]

        return {'Xs': Xs, 'Ys': Ys, 'Ms': Ms, 'Ts': Ts, 'Cntr': Cntr}

    def train(self, num_iters, in_loop, in_start):
        for it in range(num_iters):
            for _ in range(in_loop):
                end_idx = min(100 + (it + 1) * 100, init_param.K)
                Z_window = Z[:end_idx]
                data = self.smc(in_start, Z_window, cur_param)
                model.em_step(Z, data, mode=1)
                in_start += 1
            inducing_counts[it] = model.event_cnt

    # ---------------- Gamma MAP via torch LBFGS ----------------

    @staticmethod
    def map_model_fit(x, a0, b0, c0, d0) -> Tuple[float, float]:
        """
        MAP for Gamma(shape=alpha, scale=theta) with Gamma priors on alpha and theta.
        Optimize in unconstrained space and softplus back to positive.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.flatten()
        device = x.device

        N = torch.tensor(float(x.numel()), device=device)
        S = x.sum()
        L = torch.log(x).sum()

        # Unconstrained parameters
        p_alpha = torch.nn.Parameter(torch.tensor(0.0, device=device))
        p_theta = torch.nn.Parameter(torch.log(x.mean() + 1e-6))  # log-init near mean

        def sp(z):  # softplus
            return torch.nn.functional.softplus(z) + 1e-8

        opt = torch.optim.LBFGS([p_alpha, p_theta], max_iter=200, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            alpha = sp(p_alpha)
            theta = sp(p_theta)

            # Negative log posterior (same terms as original, using torch)
            # lp = ...; we minimize -lp
            lp = - N * torch.lgamma(alpha)
            lp += (alpha - 1.0) * L
            lp += - N * alpha * torch.log(theta)
            lp += - S / theta
            # priors:
            # alpha ~ Gamma(a0,b0) with scale 'b0'  (same as original)
            # theta ~ Gamma(c0,d0) with scale 'd0'
            a0_t = torch.tensor(a0, device=device)
            b0_t = torch.tensor(b0, device=device)
            c0_t = torch.tensor(c0, device=device)
            d0_t = torch.tensor(d0, device=device)
            lp += (a0_t - 1.0) * torch.log(alpha) - alpha / b0_t
            lp += (c0_t - 1.0) * torch.log(theta) - theta / d0_t

            neg = -lp
            neg.backward()
            return neg

        opt.step(closure)

        with torch.no_grad():
            alpha = torch.nn.functional.softplus(p_alpha).item()
            theta = torch.nn.functional.softplus(p_theta).item()
        return float(alpha), float(theta)

    # ---------------- EM step ----------------

    def em_step(self, Z: torch.Tensor, data: Dict[str, torch.Tensor], mode: int = 1) -> None:
        Xs, Ys, Ms, Ts, Cntr = data['Xs'], data['Ys'], data['Ms'], data['Ts'], data['Cntr']
        sample = self.sample

        gam_a0, gam_b0 = self.gam_a0, self.gam_b0
        gam_c0, gam_d0 = self.gam_c0, self.gam_d0
        norm_mu0, norm_k0 = self.norm_mu0, self.norm_k0
        norm_alpha, norm_beta = self.norm_alpha, self.norm_beta

        # X(0)
        xt = Xs[:, 0]
        if mode == 2:
            xt, idx_unique = torch.unique(xt, return_inverse=False, return_counts=False, sorted=False), None
        mx = xt.mean().item()
        new_x_mu0 = (norm_k0 * self.norm_mu0 + sample * mx) / (norm_k0 + sample)
        sx = float(((xt - new_x_mu0) ** 2).sum().item())
        new_x_var0 = (2 * norm_beta + sx + norm_k0 * (self.norm_mu0 - new_x_mu0) ** 2) / (
                    2 * norm_alpha + xt.numel() + 3)
        self.x_mu0 = new_x_mu0
        self.x_var0 = new_x_var0

        # Y(0)
        yt = Ys[:, 0]
        if mode == 2 and idx_unique is not None:
            yt = yt[idx_unique]
        my = yt.mean().item()
        new_y_mu0 = (norm_k0 * self.norm_mu0 + sample * my) / (norm_k0 + sample)
        sy = float(((yt - new_y_mu0) ** 2).sum().item())
        new_y_var0 = (2 * norm_beta + sy + norm_k0 * (self.norm_mu0 - new_y_mu0) ** 2) / (
                    2 * norm_alpha + yt.numel() + 3)
        self.y_mu0 = new_y_mu0
        self.y_var0 = new_y_var0

        # Per-event parameters
        max_events_obs = int(torch.max(Cntr).item())
        for e in range(2, max_events_obs + 1):
            idx = e - 2
            inds = torch.nonzero(Cntr >= e, as_tuple=False).squeeze()
            if inds.numel() == 0:
                continue

            # marks
            ms = Ms[inds, e - 1]
            m_mean = ms.mean().item()
            new_mark_mean = (norm_k0 * self.norm_mu0 + ms.numel() * m_mean) / (norm_k0 + ms.numel())
            s2 = float(((ms - new_mark_mean) ** 2).sum().item())
            new_mark_var = (2 * norm_beta + s2 + norm_k0 * (self.norm_mu0 - new_mark_mean) ** 2) / (
                        2 * norm_alpha + ms.numel() + 3)
            self.mark_mean[idx] = new_mark_mean
            self.mark_var[idx] = new_mark_var

            # waiting times
            ts = Ts[inds, e - 1] - Ts[inds, e - 2]
            alpha_map, theta_map = HiSDE1D.map_model_fit(ts, gam_a0, gam_b0, gam_c0, gam_d0)
            self.gam_alpha[idx] = alpha_map
            self.gam_beta[idx] = theta_map

        self.event_cnt = max_events_obs - 1

    # ---------------- synthetic Z generators ----------------

    def model_z_1d(self, model: int) -> torch.Tensor:
        K = self.K
        dt = self.dt
        z_var = self.z_var
        device = self.device

        if model == 1:
            k = torch.arange(1, K + 1, dtype=torch.float32, device=device)
            Z = torch.exp(torch.cos(2 * math.pi * 0.017 * dt * k))
            Z = Z + torch.randn(K, device=device) * math.sqrt(z_var)

        elif model == 2:
            k = torch.arange(1, K + 1, dtype=torch.float32, device=device)
            Z = torch.maximum(0.5 * dt * (k ** 1.3), torch.tensor(10.0, device=device))
            Z = Z + torch.randn(K, device=device) * math.sqrt(z_var)

        elif model == 3:
            ts = torch.arange(1, K + 1, dtype=torch.float32, device=device) * dt
            # chirp then reverse (to mirror your scipy version)
            Zc = torch_chirp_linear(ts, f0=0.001, t1=8.0, f1=0.002)
            Zc = torch.flip(Zc, dims=[0])
            Z = 7.0 * torch.exp(-Zc) + torch.randn(K, device=device) * math.sqrt(z_var)
            Z = Z - Z.mean()

        elif model == 4:
            dt_sim = 0.01

            def f(t, Y):
                sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
                x, y, z = Y.unbind(-1)
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z
                return torch.stack([dx, dy, dz], dim=-1)

            y0 = torch.tensor([1.0, -1.0, -1.0], device=device)
            traj = rk4(f, y0, 0.0, K * dt_sim, dt_sim, device=device)  # (steps+1, 3)
            z = traj[:K, 2]  # take first K points
            Z = z + torch.randn_like(z) * math.sqrt(z_var)

        else:
            raise ValueError("model must be 1, 2, 3, or 4")

        return Z


# ------------------------ GP comparison (torch-only) ------------------------

def gp_fit_compare(Z, data, init_param, cur_param, mt):
    """
    Compare GP fits using a tiny torch GP on:
      (1) uniform subsample
      (2) model-driven inducing points from inferred waiting times
    """
    K = init_param.K
    dt = init_param.dt
    z_var = init_param.z_var
    device = init_param.device

    Z_t = Z.detach().to(device).float()

    # 1) Uniform grid
    mean_interval_uni = float(torch.tensor(mt).float().mean().item())
    print(f"Overall mean interval: {mean_interval_uni:.3f} s")
    t_points = max(1, int(round(mean_interval_uni / dt)))
    print(f"t_points (steps): {t_points}")
    t_ind_uni = torch.arange(1, K + 1, t_points, device=device)
    print(f"Uniform indices: {t_ind_uni.cpu().numpy()}")
    Z_ind_uni = Z_t[t_ind_uni - 1].view(-1, 1)

    # 2) Fit GP on uniform subsample
    gp_uni = TorchGP(alpha=z_var, device=device)
    gp_uni.fit(t_ind_uni.view(-1, 1).float(), Z_ind_uni.view(-1), steps=60)
    t_full = torch.arange(1, K + 1, device=device).view(-1, 1).float()
    Z_gp_uni, Z_std_uni = gp_uni.predict(t_full, return_std=True)
    Z_gp_uni = Z_gp_uni.detach().cpu().numpy()

    # 3) Model-driven inducing points
    t_pts_model = torch.round(torch.tensor(mt, device=device, dtype=torch.float32) / dt).to(torch.long)
    t_cum = torch.cumsum(t_pts_model[:-1], dim=0)
    inds = t_cum[t_cum < K]
    t_ind_model = torch.unique(
        torch.cat([torch.tensor([1], device=device), inds, torch.tensor([K], device=device)])).to(torch.long)
    print(f"Model inducing indices: {t_ind_model.cpu().numpy()}")
    print(f"  → {t_ind_model.numel()} points, min={int(t_ind_model.min())}, max={int(t_ind_model.max())}")
    Z_ind_model = Z_t[t_ind_model - 1].view(-1, 1)

    # 4) Fit GP on model-driven subsample
    gp_model = TorchGP(alpha=z_var, device=device)
    gp_model.fit(t_ind_model.view(-1, 1).float(), Z_ind_model.view(-1), steps=60)
    Z_gp_model, Z_std_model = gp_model.predict(t_full, return_std=True)
    Z_gp_model = Z_gp_model.detach().cpu().numpy()

    # 5) MSEs
    Z_np = Z_t.cpu().numpy().ravel()
    mse_uni = float(((torch.tensor(Z_np) - torch.tensor(Z_gp_uni)) ** 2).mean())
    mse_model = float(((torch.tensor(Z_np) - torch.tensor(Z_gp_model)) ** 2).mean())
    print(f"Uniform grid MSE: {mse_uni:.4f}")
    print(f"Model-driven MSE: {mse_model:.4f}")

    # 6) Mean latent Y
    Y_np = data['Ys'][:, :K].detach().cpu().numpy()
    meanY = Y_np.mean(axis=0)

    # 7) Plot (matplotlib)
    fig, axes = plt.subplots(4, 1, figsize=(6, 8))
    t_full_np = t_full.cpu().numpy()

    # Panel 1: Observed Z
    axes[0].plot(t_full_np, Z_np, linewidth=1)
    axes[0].set(xlabel='Time index', ylabel='Z', title='Observed signal')

    # Panel 2: Mean latent increment X
    X_np = data['Xs'].detach().cpu().numpy().mean(axis=0)
    tX = torch.arange(1, len(X_np) + 1).numpy()
    axes[1].plot(tX, X_np, linewidth=1.5)
    axes[1].set_xlim(1, K)
    axes[1].set(xlabel='Time index', ylabel='X', title='Mean latent increment')

    # Panel 3: Marks vs. inferred times
    Ts_np = data['Ts'].detach().cpu().numpy()
    Ms_np = data['Ms'].detach().cpu().numpy()
    Cntr_np = data['Cntr'].detach().cpu().numpy().astype(int)
    for s in range(init_param.sample):
        cnt = Cntr_np[s]
        if cnt > 0:
            tau_idx = Ts_np[s, :cnt] / dt
            axes[2].plot(tau_idx, Ms_np[s, :cnt], linewidth=1)
    axes[2].set_xlim(1, K)
    axes[2].set(xlabel='Time index', ylabel='Mark m', title='Marks vs. inferred times')

    # Panel 4: GP fits comparison
    axes[3].plot(t_full_np, Z_np, linewidth=1.5)
    axes[3].plot(t_full_np, Z_gp_uni, linestyle='--', linewidth=1.5)
    axes[3].plot(t_full_np, meanY, linestyle=':', linewidth=1.5)
    axes[3].set(xlabel='Time index', title='GP fits comparison')
    axes[3].legend(['Z', 'GP_uniform', 'Latent-mean'], loc='upper right')

    plt.tight_layout()
    plt.show()


# ------------------------------- main / demo -------------------------------

if __name__ == "__main__":
    # match random seeds
    torch.manual_seed(0)

    # Inference parameters
    in_loop = 2
    num_iters = 6
    in_start = 1
    inducing_counts = torch.zeros(num_iters, dtype=torch.long)

    # Initialize model
    model = HiSDE1D()
    init_param = model
    cur_param = model

    # Generate synthetic observations
    Z = model.model_z_1d(3)

    # Inference loop
    for it in range(num_iters):
        print(f"[EM] iteration {it+1}/{num_iters}:")
        for _ in range(in_loop):
            end_idx = min(100 + (it + 1) * 100, init_param.K)
            Z_window = Z[:end_idx]
            data = model.smc_fast(in_start, Z_window, cur_param)
            model.em_step(Z, data, mode=1)
            in_start += 1
        inducing_counts[it] = model.event_cnt

    # Final waiting-time stats
    gam_alpha = model.gam_alpha[:model.event_cnt].detach().cpu().numpy()
    gam_beta = model.gam_beta[:model.event_cnt].detach().cpu().numpy()
    mt = gam_alpha * gam_beta
    avg_wait = float(torch.tensor(mt).float().mean())

    print(f"Initial inducing-point count: {int(inducing_counts[0])}")
    print(f"Final inducing-point count:   {int(inducing_counts[-1])}")
    print(f"Average waiting time:         {avg_wait:.3f} time-steps")

    # Plot GP-comparison
    gp_fit_compare(Z, data, init_param, cur_param, mt)
