# src/models/model.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
from tqdm import tqdm


class HierarchicalSDEParticleFilter(nn.Module):
    """
    Hierarchical SDE with inducing points and SMC/EM training.
    Structure only (Step 2) — core logic lands in Steps 3–5.

    Components (with references):
      - Inducing-point priors (Gamma for waiting times, Gaussian for marks)
          -> Renewal marked point process  [Eq. (2), Sec. \\ref{hsde}]
      - First SDE layer X (Brownian-bridge-like dynamics between events)
          -> Continuous eqs in text; discrete-time update in  [Sec. \\ref{a33}]
      - Second SDE layer Y (integrator driven by X)
          -> Discrete-time update in  [Sec. \\ref{a33}]
      - Observation Z = W Y + noise
          ->  [Eq. \\ref{eq6}]
      - SMC/EM scaffolding
          ->  [Alg. \\ref{alg1}, Alg. \\ref{alg2}, Sec. \\ref{inference}, \\ref{a23}, \\ref{a43}]
    """

    def __init__(
            self,
            M: int,  # observation dim
            D_latent: int,  # latent dim (for X_t, Y_t)
            dt: float,  # sampling interval Delta t
            num_steps: int,  # K
            device: torch.device = torch.device("cpu"),
            config: Optional[Dict] = None,
            sigma_x=1e-1,
            sigma_y=1e-4,
            k_tau=20,
            lmb_tau=0.5
    ):
        super().__init__()
        self.M = int(M)
        self.D = int(D_latent)
        self.dt = float(dt)
        self.K = int(num_steps)
        self.device = device
        self.config = config or {}

        # -------------------------
        # Observation model params [Eq. (6)]
        # Z_k = W Y_k + eps,  eps ~ N(0, R)
        # -------------------------
        # W in R^{M x D}
        self.W = nn.Parameter(torch.randn(self.M, self.D) * 0.05)
        # We keep R diagonal initially; store as log-variances for positivity
        self.log_R_diag = nn.Parameter(torch.log(torch.ones(self.M) * 1e-1))

        # -------------------------
        # SDE noise scales (shared across dims by default)
        # X-layer diffusion scale (sigma_x) and Y-layer diffusion scale (sigma_y)
        # See SDE defs and discrete forms in [Sec. \\ref{a33}]
        # -------------------------
        self.log_sigma_x = nn.Parameter(torch.log(torch.tensor(sigma_x)))  # to be set in Step 3
        self.log_sigma_y = nn.Parameter(torch.log(torch.tensor(sigma_y)))  # to be set in Step 3

        # -------------------------
        # Inducing-point priors (renewal marked point process) [Eq. (2), Sec. \\ref{hsde}]
        # tau_i ~ Gamma(shape=k_tau, rate=lambda_tau)
        # m_i  ~ N(mu_m, Sigma_m)
        # We'll start with diagonal Sigma_m (store log-vars); can generalize later.
        # -------------------------

        self.k_tau = nn.Parameter(torch.tensor(k_tau))  # shape (will be set in Step 3)
        self.lmb_tau = nn.Parameter(torch.tensor(lmb_tau))  # rate (Step 3 ensures mean ~40s)
        self.mu_m = nn.Parameter(torch.zeros(self.D))  # D-dim mean for marks
        self.log_var_m = nn.Parameter(torch.full((self.D,), 0.0))  # diagonal log-variance

        # -------------------------
        # Initial state priors p(x0), p(y0) (Gaussian)
        # -------------------------
        self.x0_mean = nn.Parameter(torch.zeros(self.D))
        self.x0_log_var = nn.Parameter(torch.full((self.D,), math.log(1e-2)))
        self.y0_mean = nn.Parameter(torch.zeros(self.D))
        self.y0_log_var = nn.Parameter(torch.full((self.D,), math.log(1e-2)))

        # -------------------------
        # Buffers for time grid, used in discrete updates [Sec. \\ref{a33}]
        # -------------------------
        t_grid = torch.arange(self.K, dtype=torch.float32) * self.dt
        self.register_buffer("t_grid", t_grid, persistent=False)

        # Placeholders we’ll fill during Step 4–5:
        self.initialized = True

    # ---------------------------------------------------------------------
    # Distributions for priors (used in SMC sampling and EM updates)
    #   [Eq. (2): p(tau_i) Gamma, p(m_i) Gaussian]
    # ---------------------------------------------------------------------
    def tau_prior(self) -> D.Gamma:
        # Gamma(shape=k, rate=lambda)  (PyTorch uses concentration=alpha, rate=beta)
        return D.Gamma(
            concentration=self.k_tau.exp() if self.config.get("exp_param", False) else self.k_tau.clamp_min(1e-5),
            rate=self.lmb_tau.clamp_min(1e-8))

    def mark_prior(self) -> D.Independent:
        var = self.log_var_m.exp()
        return D.Independent(D.Normal(loc=self.mu_m, scale=var.sqrt()), 1)

    def x0_prior(self) -> D.Independent:
        var = self.x0_log_var.exp()
        return D.Independent(D.Normal(self.x0_mean, var.sqrt()), 1)

    def y0_prior(self) -> D.Independent:
        var = self.y0_log_var.exp()
        return D.Independent(D.Normal(self.y0_mean, var.sqrt()), 1)

    # ---------------------------------------------------------------------
    # Discrete-time dynamics [Sec. \\ref{a33}]
    #   X_{k+1} = X_k + ((m_{i+1} - X_k)/(t_{i+1} - k*dt)) * dt
    #             + sqrt( ((t_{i+1} - k*dt)*(k*dt - t_i)) / (t_{i+1}-t_i) * dt ) * w_k
    #   with  w_k ~ N(0, sigma_x^2 I)
    # ---------------------------------------------------------------------
    def x_step(
            self,
            x_k: torch.Tensor,  # (..., D)
            m_next: torch.Tensor,  # (..., D) mark for segment’s right endpoint
            t_left: torch.Tensor,  # (...,)  scalar t_i
            t_right: torch.Tensor,  # (...,)  scalar t_{i+1}
            t_k: torch.Tensor,  # (...,)  scalar current time k*dt
            rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # [Sec. \ref{a33}] Brownian-bridge-like discrete update for X
        dt = self.dt
        denom = (t_right - t_k).clamp_min(1e-8)  # (avoid divide-by-zero)
        drift = (m_next - x_k) / denom  # (D)
        # base bridge variance factor
        num_var = (t_right - t_k) * (t_k - t_left)
        base = (num_var / (t_right - t_left).clamp_min(1e-8)).clamp_min(0.0) * dt  # scalar per batch
        sigma_x = self.log_sigma_x.exp()
        noise_std = torch.sqrt((sigma_x ** 2) * base)  # (..., 1)
        eps = torch.zeros_like(x_k).normal_(mean=0.0, std=1.0, generator=rng) * noise_std
        x_next = x_k + drift * dt + eps
        return x_next

    # ---------------------------------------------------------------------
    # Y-layer integrator [Sec. \\ref{a33}]
    #   Y_{k+1} = Y_k + X_k * dt + sqrt(dt) * nu_k,  nu_k ~ N(0, sigma_y^2 I)
    # ---------------------------------------------------------------------
    def y_step(
            self,
            y_k: torch.Tensor,  # (..., D)
            x_k: torch.Tensor,  # (..., D)
            rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        dt = self.dt
        sigma_y = self.log_sigma_y.exp()
        eps = torch.zeros_like(y_k).normal_(mean=0.0, std=1.0, generator=rng) * (sigma_y * math.sqrt(dt))
        y_next = y_k + x_k * dt + eps
        return y_next

    # ---------------------------------------------------------------------
    # Observation model and likelihood [Eq. (6)]
    #   Z_k | Y_k ~ N(W Y_k, R)
    # ---------------------------------------------------------------------
    def obs_mean(self, y_k: torch.Tensor) -> torch.Tensor:
        return y_k @ self.W.t()  # (.., D) x (D,M)^T -> (.., M)

    def obs_dist(self, y_k: torch.Tensor) -> D.Independent:
        mean = self.obs_mean(y_k)
        R_diag = self.log_R_diag.exp().clamp_min(1e-10)
        return D.Independent(D.Normal(loc=mean, scale=R_diag.sqrt()), 1)

    def apply_paper_initialization(
            self,
            obs_noise_variance: float,
            mean_wait_time_sec: float = 40.0,
            std_wait_time_sec: float = 8.94,
            sigma_x: float = 1e-1,
            sigma_y: float = 1e-4,
            num_particles: int = 10_000,
            em_iters: int = 20,
            fix_sigma_y: bool = True,
    ) -> None:
        """
        Initialize parameters and training hyperparams as described in the paper.

        Args:
            obs_noise_variance: known observation noise variance (diagonal R) [Eq. \\ref{eq6}]
            mean_wait_time_sec: Gamma mean for waiting times tau_i            [Eq. (2)]
            std_wait_time_sec:  Gamma std for waiting times tau_i             [Eq. (2)]
            sigma_x: diffusion scale for X-layer SDE                          [Sec. \\ref{a33}]
            sigma_y: diffusion scale for Y-layer SDE                          [Sec. \\ref{a33}]
            num_particles: SMC particle count                                 [Alg. \\ref{alg1}, \\ref{alg2}]
            em_iters: number of EM iterations                                 [Sec. \\ref{inference}, \\ref{a43}]
            fix_sigma_y: keep sigma_y fixed during training (paper rationale in \\ref{a43})
        """
        # ---- Observation noise R (diagonal)  [Eq. \ref{eq6}] ----
        # R = diag(obs_noise_variance)  (assumed known in the paper)
        with torch.no_grad():
            self.log_R_diag.copy_(
                torch.log(torch.full_like(self.log_R_diag, float(obs_noise_variance)).clamp_min(1e-12)))

        # ---- Waiting times Gamma(k, lambda)  [Eq. (2), Sec. \ref{hsde}] ----
        # For Gamma(shape=k, rate=λ): mean = k/λ,  std = sqrt(k)/λ
        # Solve from mean, std: k = (mean/std)^2,  λ = k/mean
        k = (mean_wait_time_sec / std_wait_time_sec) ** 2
        lam = k / mean_wait_time_sec
        with torch.no_grad():
            self.k_tau.copy_(torch.tensor(k, dtype=self.k_tau.dtype, device=self.k_tau.device))
            self.lmb_tau.copy_(torch.tensor(lam, dtype=self.lmb_tau.dtype, device=self.lmb_tau.device))

        # (Optional sanity check values as buffers for logging/debug — not used in math)
        self.register_buffer("paper_tau_mean_init", torch.tensor(mean_wait_time_sec, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("paper_tau_std_init", torch.tensor(std_wait_time_sec, dtype=torch.float32),
                             persistent=False)

        # ---- SDE noise scales  [Sec. \ref{a33}] ----
        # Brownian-bridge layer (X): sigma_x
        # Integrator layer (Y): sigma_y (often fixed to allow info flow X->Y; see Sec. \ref{a43})
        with torch.no_grad():
            self.log_sigma_x.copy_(torch.log(torch.tensor(float(sigma_x), device=self.log_sigma_x.device)))
            self.log_sigma_y.copy_(torch.log(torch.tensor(float(sigma_y), device=self.log_sigma_y.device)))

        # Optionally freeze sigma_y per paper note in [Sec. \ref{a43}]
        for p in [self.log_sigma_y] if fix_sigma_y else []:
            p.requires_grad = False

        # ---- Training hyperparameters ----
        # SMC particles and EM iterations [Alg. \ref{alg1}, \ref{alg2}, Sec. \ref{inference}]
        self.config["num_particles"] = int(num_particles)
        self.config["em_iters"] = int(em_iters)

        # In practice, ensure Δt << min(tau_i) [Sec. \ref{a33}] — we can't enforce here,
        # but you can check posterior samples of tau_i later and adjust dt if needed.
        self.config.setdefault("ensure_dt_condition_note", "Make sure Δt << min(tau) as in Sec. \\ref{a33}.")

    # Convenience getters (optional)
    @property
    def sigma_x(self) -> torch.Tensor:
        return self.log_sigma_x.exp()

    @property
    def sigma_y(self) -> torch.Tensor:
        return self.log_sigma_y.exp()

    @property
    def R_diag(self) -> torch.Tensor:
        return self.log_R_diag.exp()

    @torch.no_grad()
    def smc(
            self,
            Z: torch.Tensor,  # (K, M)
            num_particles: Optional[int] = None,
            ess_fraction: float = 0.5,  # resample when ESS < ess_fraction * U
            seed: Optional[int] = None,
    ) -> Dict:
        device = Z.device
        K, M = Z.shape
        assert K == self.K and M == self.M, "Shape mismatch with model setup."

        U = int(num_particles or self.config.get("num_particles", 10000))
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)

        # --- 1) Initialize particles  [Alg. \ref{alg2} Initialization] ---
        x = self.x0_prior().sample((U,))  # (U, D)
        y = self.y0_prior().sample((U,))  # (U, D)
        w = torch.full((U,), 1.0 / U, device=device)

        # Inducing-point state per particle:
        # For each particle u, maintain current [t_i, t_{i+1}] and m_{i+1}.
        t_left = torch.zeros(U, device=device)  # t_i
        tau = self.tau_prior().sample((U,))  # tau_1  [Eq. (2)]
        t_right = t_left + tau  # t_{i+1}
        m_next = self.mark_prior().sample((U,))  # m_{i+1}  [Eq. (2)]
        # Count of events per particle (for stats)
        n_events = torch.zeros(U, dtype=torch.long, device=device)

        # --- storage for E-step expectations ---
        # E[Y_k] and E[Y_k Y_k^T] via particle average (after resampling or with weights)
        EY = torch.zeros(K, self.D, device=device)
        EYYt = torch.zeros(K, self.D, self.D, device=device)

        # Optionally collect inducing samples (weighted) for MAP updates
        collected_tau = []
        collected_m = []

        # simple multinomial resampling helper
        def resample_multinomial(x_like, w):
            idx = torch.multinomial(w, num_samples=U, replacement=True, generator=gen)
            return idx

        # --- 2) Sequential update over k = 0..K-1  [Alg. \ref{alg1}/\ref{alg2}] ---
        for k in tqdm(range(K), desc="Sequential update over k = 0...K-1"):
            t_k = self.t_grid[k]  # k * dt

            # (A) If we passed right boundary, sample a new inducing point [Alg. step 1]
            need_new = t_k > t_right
            if need_new.any():
                # Shift segment: t_i <- t_{i+1}, draw new tau & mark
                t_left = torch.where(need_new, t_right, t_left)
                tau_new = self.tau_prior().sample((U,))
                if len(tau_new.shape) == 1:
                    tau_new = tau_new.unsqueeze(1)
                m_new = self.mark_prior().sample((U,))
                t_right = torch.where(need_new, t_left + tau_new, t_right)
                m_next = torch.where(need_new, m_new, m_next)
                n_events = n_events.unsqueeze(1) + need_new.long()
                # Collect samples (for prior updates)
                collected_tau.append(tau_new[need_new])
                collected_m.append(m_new[need_new])

            if len(t_right.shape) == 1:
                t_right = t_right.unsqueeze(1)
            if len(t_left.shape) == 1:
                t_left = t_left.unsqueeze(1)
            # (B) Propagate X, then Y  [Sec. \ref{a33}]
            x = self.x_step(
                x_k=x, m_next=m_next,
                t_left=t_left, t_right=t_right, t_k=t_k,
                rng=gen,
            )
            y = self.y_step(y_k=y, x_k=x, rng=gen)

            # (C) Weights from observation likelihood (bootstrap proposal)
            #     p(z_k | y_k)  [Eq. \ref{eq6}]
            obs = self.obs_dist(y)  # N(W y, R)
            log_w_inc = obs.log_prob(Z[k].expand_as(obs.base_dist.loc))  # (U,)
            # normalize in log-space for stability
            maxlw = log_w_inc.max()
            w = w * torch.exp(log_w_inc - maxlw)
            w = w / w.sum().clamp_min(1e-12)

            # (D) Compute ESS; resample if needed  [Alg. step 4]
            ess = 1.0 / (w.pow(2).sum().clamp_min(1e-12))
            if ess < ess_fraction * U:
                idx = resample_multinomial(y, w)
                x = x[idx]
                y = y[idx]
                t_left = t_left[idx]
                t_right = t_right[idx]
                m_next = m_next[idx]
                n_events = n_events[idx]
                w.fill_(1.0 / U)

            # (E) Accumulate expectations (after normalization / resampling)
            #     E[Y_k] ≈ sum_u w_u y_k^u , E[Y_k Y_k^T] ≈ sum_u w_u y_k^u y_k^{uT}
            EY[k] = torch.einsum('u,ud->d', w, y)  # (D,)
            EYYt[k] = torch.einsum('u,ud,ue->de', w, y, y)  # (D, D)
            # Gather inducing samples (concat) for M-step prior updates

        if len(collected_tau) == 0:
            tau_all = torch.empty(0, device=device)
            m_all = torch.empty(0, self.D, device=device)
        else:
            tau_all = torch.cat(collected_tau, dim=0)
            m_all = torch.cat(collected_m, dim=0)

        return {
            "EY": EY,  # (K, D)
            "EYYt": EYYt,  # (K, D, D)
            "last_weights": w,  # (U,)
            "n_events": n_events,  # (U,)
            "tau_samples": tau_all,  # (~N_tau,)
            "m_samples": m_all,  # (~N_tau, D)
        }

    # ------------------------------------------------------------
    # EM epoch: E-step via SMC, then M-step parameter updates
    # References:
    #   - Objective/likelihood factorization: [Sec. \ref{a43}, Eq. \ref{eq17}]
    #   - Observation updates (W,R):          [Eq. \ref{eq6}]
    #   - Priors for tau,m:                   [Eq. (2)]
    #   - Sigma updates:                      [Sec. \ref{a33}]  (sigma_y often fixed per \ref{a43})
    # ------------------------------------------------------------
    @torch.no_grad()
    def em(self, Z: torch.Tensor, seed: Optional[int] = None) -> Dict:
        out = self.smc(Z, seed=seed)

        EY = out["EY"]  # (K, D)
        EYYt = out["EYYt"]  # (K, D, D)
        tau_samples = out["tau_samples"]  # (~N_tau,)
        m_samples = out["m_samples"]  # (~N_tau, D)

        # ---------- M-step: Observation parameters W, R  [Eq. \ref{eq6}] ----------
        # Minimize sum_k ||Z_k - W E[Y_k]||_R^2 -> GLS; here we use closed-form with diag R.
        # Sufficient statistics:
        #   S_y  = sum_k E[Y_k Y_k^T]
        #   S_zy = sum_k Z_k E[Y_k]^T
        Sy = EYYt.sum(dim=0)  # (D, D)
        Szy = torch.einsum("km,kd->md", Z, EY)  # (M, D)
        # Regularize Sy (in case of rank issues early on)
        Sy_reg = Sy + 1e-6 * torch.eye(self.D, device=Sy.device)
        W_new = Szy @ torch.linalg.inv(Sy_reg)
        self.W.copy_(W_new)

        # R diagonal from residuals:
        #   R = (1/K) sum_k E[(Z_k - W Y_k)(Z_k - W Y_k)^T]
        # Approximate with point E[Y_k]
        resid = Z - EY @ self.W.t()  # (K, M)
        R_diag_new = (resid.pow(2).mean(dim=0)).clamp_min(1e-10)
        self.log_R_diag.copy_(torch.log(R_diag_new))

        # ---------- M-step: Priors p(tau), p(m)  [Eq. (2)] ----------
        # Moment-matched MAP (no strong hyperprior here):
        if tau_samples.numel() >= 5:  # need some samples
            mean_tau = tau_samples.mean()
            std_tau = tau_samples.std(unbiased=False).clamp_min(1e-8)
            k = (mean_tau / std_tau) ** 2  # mean=k/lambda, std=sqrt(k)/lambda
            lam = k / mean_tau
            self.k_tau.copy_(k)
            self.lmb_tau.copy_(lam)

        if m_samples.numel() >= self.D:
            mu = m_samples.mean(dim=0)  # (D,)
            var = m_samples.var(dim=0, unbiased=False).clamp_min(1e-8)
            self.mu_m.copy_(mu)
            self.log_var_m.copy_(torch.log(var))

        # ---------- (Optional) sigma_x update  [Sec. \ref{a33}] ----------
        # A simple moment estimator from bridge residuals; for stability we only shrink mildly.
        # NOTE: Precise closed-form depends on full E[r_k^T r_k / base_k]; here we proxy via
        # residual energy in observation and a small trust region — practical and stable.
        if self.config.get("learn_sigma_x", True):
            # crude proxy: if residual is large, allow sigma_x to increase slightly
            scale_proxy = resid.pow(2).mean().sqrt().item()
            curr = float(self.sigma_x)
            target = max(curr * 0.9, min(curr * 1.1, 0.1 * scale_proxy + 1e-3))
            self.log_sigma_x.copy_(torch.log(torch.tensor(target, device=self.log_sigma_x.device)))

        return {
            "W": self.W.detach().clone(),
            "R_diag": self.log_R_diag.exp().detach().clone(),
            "k_tau": float(self.k_tau),
            "lmb_tau": float(self.lmb_tau),
            "mu_m": self.mu_m.detach().clone(),
            "var_m": self.log_var_m.exp().detach().clone(),
        }

    # ------------------------------------------------------------
    # High-level training loop (EM over several iterations)
    # References:
    #   - EM setup: [Sec. \ref{inference}, \ref{a43}]
    #   - SMC used in the E-step: [Alg. \ref{alg1}, \ref{alg2}]
    # ------------------------------------------------------------
    def fit(
        self,
        Z: torch.Tensor,
        max_iter: Optional[int] = None,
        seed: Optional[int] = None,
        eval_every: int = 1,
        eval_kwargs: Optional[Dict] = None,   # e.g., {"num_particles": 5000, "ess_fraction": 0.5}
        clean: Optional[torch.Tensor] = None, # optional clean target for MSE
        save_best_path: Optional[str] = None, # if set, save best-by-log_marginal
    ) -> Dict:
        Z = Z.to(next(self.parameters()).device)
        if clean is not None:
            clean = clean.to(Z.device)
        iters = int(max_iter or self.config.get("em_iters", 20))
        eval_kwargs = eval_kwargs or {}

        history = []
        best_score = -float("inf")

        for it in range(iters):
            # ---- EM epoch ----
            stats = self.em(Z, seed=None if seed is None else seed + it)

            # ---- Evaluation (every eval_every iters) ----
            metrics = None
            post = None
            if (it + 1) % max(1, eval_every) == 0:
                eval_out = self.evaluate(Z, clean=clean, **eval_kwargs)
                metrics = eval_out["metrics"]
                post = eval_out["post"]
                print(f"[EM] iteration {it+1}/{iters}:"
                      f"mse_observed = {np.round(metrics['mse_observed'], 2)} \t "
                      f"mse_vs_clean = {np.round(metrics['mse_vs_clean'], 2)} \t "
                      f"log_marginal = {np.round(metrics['log_marginal'], 2)} \t ")
                # track best by log-marginal (higher is better) [Eq. \ref{eq17}]
                score = metrics.get("log_marginal", float("-inf"))
                if save_best_path and score > best_score:
                    best_score = score
                    self.save(save_best_path, extra={"iter": it + 1, "metrics": metrics})

            # ---- Append a rich snapshot to history ----
            history.append({
                "iter": it + 1,
                "params": {
                    "W": self.W.detach().cpu().clone(),
                    "R_diag": self.log_R_diag.exp().detach().cpu().clone(),
                    "k_tau": float(self.k_tau),
                    "lmb_tau": float(self.lmb_tau),
                    "mu_m": self.mu_m.detach().cpu().clone(),
                    "var_m": self.log_var_m.exp().detach().cpu().clone(),
                    "sigma_x": float(self.sigma_x),
                    "sigma_y": float(self.sigma_y),
                },
                "metrics": metrics,  # may be None on non-eval steps
            })

        return {"history": history}

    # ------------------------------------------------------------
    # Inference: posterior summaries using SMC with learned params
    # References:
    #   - Algorithmic flow: [Alg. \ref{alg1}, \ref{alg2}]
    #   - Dynamics (X,Y):   [Sec. \ref{a33}]
    #   - Observation:      [Eq. \ref{eq6}]
    #   - Predictive log-lik via PF normalization at each step
    # ------------------------------------------------------------
    @torch.no_grad()
    def infer(
            self,
            Z: torch.Tensor,  # (K, M)
            num_particles: Optional[int] = None,
            ess_fraction: float = 0.5,
            seed: Optional[int] = None,
            return_particles: bool = False,
    ) -> Dict:
        device = Z.device
        K, M = Z.shape
        assert K == self.K and M == self.M, "Shape mismatch with model setup."

        U = int(num_particles or self.config.get("num_particles", 10_000))
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)

        # --- Initialization (as in SMC) [Alg. \ref{alg2} Initialization] ---
        x = self.x0_prior().sample((U,))  # (U, D)
        y = self.y0_prior().sample((U,))  # (U, D)
        w = torch.full((U,), 1.0 / U, device=device)

        t_left = torch.zeros(U, device=device)
        tau = self.tau_prior().sample((U,))  # [Eq. (2)]
        t_right = t_left + tau
        m_next = self.mark_prior().sample((U,))  # [Eq. (2)]
        n_events = torch.zeros(U, dtype=torch.long, device=device)

        EY = torch.zeros(K, self.D, device=device)
        EYYt = torch.zeros(K, self.D, self.D, device=device)
        Zhat = torch.zeros(K, self.M, device=device)

        # Track the PF predictive log-likelihood:
        #   log p(Z_{1:K}) = sum_k log( sum_u w_{k-1} * p(z_k | y_k^u) )
        # This uses bootstrap proposal; see weight update in [Alg. \ref{alg1}].
        log_marginal = 0.0

        # Optional particle storage (can be heavy)
        X_store = [] if return_particles else None
        Y_store = [] if return_particles else None
        W_store = [] if return_particles else None

        def resample_multinomial(w):
            idx = torch.multinomial(w, num_samples=U, replacement=True, generator=gen)
            return idx

        for k in range(K):
            t_k = self.t_grid[k]

            # New inducing-point segment if we crossed t_right  [Alg. step 1]
            need_new = t_k > t_right
            if need_new.any():
                t_left = torch.where(need_new, t_right, t_left)
                tau_new = self.tau_prior().sample((U,))
                if len(tau_new.shape) == 1:
                    tau_new = tau_new.unsqueeze(1)
                m_new = self.mark_prior().sample((U,))
                t_right = torch.where(need_new, t_left + tau_new, t_right)
                m_next = torch.where(need_new, m_new, m_next)
                n_events = n_events.unsqueeze(1) + need_new.long()

            # Propagate X then Y  [Sec. \ref{a33}]
            if len(t_right.shape) == 1:
                t_right = t_right.unsqueeze(1)
            if len(t_left.shape) == 1:
                t_left = t_left.unsqueeze(1)

            x = self.x_step(x_k=x, m_next=m_next, t_left=t_left, t_right=t_right, t_k=t_k, rng=gen)
            y = self.y_step(y_k=y, x_k=x, rng=gen)

            # Predictive likelihood at step k (before weight renorm)
            obs = self.obs_dist(y)  # N(W y, R) [Eq. \ref{eq6}]
            log_w_inc = obs.log_prob(Z[k].expand_as(obs.base_dist.loc))  # (U,)
            # predictive normalization term:
            #   p(z_k | z_{1:k-1}) = sum_u w_{k-1} * exp(log_w_inc_u)
            # compute in log-space
            maxlw = log_w_inc.max()
            pred = (w * torch.exp(log_w_inc - maxlw)).sum().clamp_min(1e-12)
            log_pred_k = float(maxlw + torch.log(pred))
            log_marginal += log_pred_k

            # Update and renormalize weights [Alg. step 3]
            w = w * torch.exp(log_w_inc - maxlw)
            w = w / w.sum().clamp_min(1e-12)

            # Resample if ESS is small [Alg. step 4]
            ess = 1.0 / (w.pow(2).sum().clamp_min(1e-12))
            if ess < ess_fraction * U:
                idx = resample_multinomial(w)
                x = x[idx]
                y = y[idx]
                t_left = t_left[idx]
                t_right = t_right[idx]
                m_next = m_next[idx]
                n_events = n_events[idx]
                w.fill_(1.0 / U)

            # Posterior expectations for Y_k  (using current weights)
            #EY[k] = (w.unsqueeze(-1) * y).sum(0)
            #EYYt[k] = (w.unsqueeze(-1) * (y.unsqueeze(-1) @ y.unsqueeze(-2))).sum(0)
            EY[k] = torch.einsum('u,ud->d', w, y)  # (D,)
            EYYt[k] = torch.einsum('u,ud,ue->de', w, y, y)  # (D, D)
            Zhat[k] = self.obs_mean(EY[k])  # W E[Y_k]  [Eq. \ref{eq6}]

            if return_particles:
                X_store.append(x.clone())
                Y_store.append(y.clone())
                W_store.append(w.clone())

        out = {
            "EY": EY,  # (K, D)
            "EYYt": EYYt,  # (K, D, D)
            "Zhat": Zhat,  # (K, M)
            "log_marginal": log_marginal,  # approx log p(Z_{1:K})
            "final_weights": w,  # (U,)
            "n_events": n_events,  # (U,)
        }
        if return_particles:
            out["particles_X"] = torch.stack(X_store, dim=0)  # (K, U, D)
            out["particles_Y"] = torch.stack(Y_store, dim=0)  # (K, U, D)
            out["particles_w"] = torch.stack(W_store, dim=0)  # (K, U)

        return out

    # ------------------------------------------------------------
    # Evaluate: simple metrics on inference results
    # References:
    #   - Reconstruction via observation model [Eq. \ref{eq6}]
    #   - PF marginal likelihood as a scoring proxy
    # ------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
            self,
            Z: torch.Tensor,  # (K, M)
            clean: Optional[torch.Tensor] = None,  # (K,) or (K,M) optional target for MSE
            **infer_kwargs,
    ) -> Dict:
        res = self.infer(Z, **infer_kwargs)
        Zhat = res["Zhat"]  # (K, M)

        # Reconstruction error vs observed Z
        mse_Z = torch.mean((Z - Zhat).pow(2)).item()

        metrics = {
            "mse_observed": mse_Z,
            "log_marginal": res["log_marginal"],  # higher is better
        }

        # Optional: if user passes a denoised/clean target
        if clean is not None:
            if clean.ndim == 1:
                clean_ = clean.unsqueeze(-1)
            else:
                clean_ = clean
            mse_clean = torch.mean((clean_ - Zhat).pow(2)).item()
            metrics["mse_vs_clean"] = mse_clean

        return {
            "metrics": metrics,
            "post": {
                "EY": res["EY"],
                "EYYt": res["EYYt"],
                "Zhat": Zhat,
                "final_weights": res["final_weights"],
            },
        }
    # ------------------------------------------------------------
    # Saving / Loading
    # References:
    #   - Saves all learned params that appear in the likelihood pieces:
    #     W, R [Eq. \ref{eq6}], sigma_x, sigma_y [Sec. \ref{a33}],
    #     Gamma/Normal priors for inducing points [Eq. (2)]
    # ------------------------------------------------------------
    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        payload = {
            "meta": {
                "M": self.M,
                "D": self.D,
                "dt": self.dt,
                "K": self.K,
                "class": self.__class__.__name__,
                "torch_version": torch.__version__,
            },
            "config": self.config,
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = "cpu") -> "HierarchicalSDEParticleFilter":
        ckpt = torch.load(path, map_location=map_location)
        meta = ckpt["meta"]
        model = cls(
            M=meta["M"],
            D_latent=meta["D"],
            dt=meta["dt"],
            num_steps=meta["K"],
            device=torch.device(map_location if isinstance(map_location, str) else "cpu"),
            config=ckpt.get("config", {}),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model