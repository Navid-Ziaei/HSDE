from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import solve_ivp
from scipy.signal import chirp


class SyntheticDataGenerator:
    """
    Generate synthetic 1D observations for various HiSDE models.

    Models:
        - Exponentiated cosine wave
        - Power-law trend with floor
        - Reversed chirp with exponential transform
        - Lorenz attractor's third coordinate
    """

    def __init__(self, num_steps: int, time_step: float, noise_variance: float, random_seed: int = None):
        """
        Parameters
        ----------
        num_steps : int
            Total number of time samples.
        time_step : float
            Time increment (Δt) between samples.
        noise_variance : float
            Variance of additive Gaussian noise.
        random_seed : int, optional
            Seed for reproducibility. If None, results will vary each run.
        """
        self.num_steps = num_steps
        self.time_step = time_step
        self.noise_variance = noise_variance
        self.rng = np.random.default_rng(random_seed)  # reproducible RNG

    # ----------------------------
    # Model 1: Exponentiated cosine
    # ----------------------------
    def generate_exponentiated_cosine(self) -> np.ndarray:
        k_vals = np.arange(1, self.num_steps + 1)
        signal = np.exp(np.cos(2 * np.pi * 0.017 * self.time_step * k_vals))
        noisy_signal = signal + np.sqrt(self.noise_variance) * self.rng.standard_normal(len(signal))
        return noisy_signal

    # -------------------------
    # Model 2: Power-law growth
    # -------------------------
    def generate_power_law(self) -> np.ndarray:
        k_vals = np.arange(1, self.num_steps + 1)
        signal = np.maximum(self.time_step * (k_vals ** 1.3) * 0.5, 10)
        noisy_signal = signal + np.sqrt(self.noise_variance) * self.rng.standard_normal(len(signal))
        return noisy_signal

    # -----------------------------
    # Model 3: Reversed chirp signal
    # -----------------------------
    def generate_reversed_chirp(self) -> np.ndarray:
        time_series = self.time_step * np.arange(1, self.num_steps + 1)
        signal = chirp(time_series, f0=0.001, t1=8, f1=0.002)  # linear chirp
        signal = signal[::-1]  # reverse in time
        transformed = 7 * np.exp(-signal)
        transformed = transformed - np.mean(transformed),
        observation = transformed + np.sqrt(self.noise_variance) * self.rng.standard_normal(len(signal))

        data = {
            'observation_z': observation,
            'latent': transformed,
            't': time_series
        }

        return data

    # --------------------------------------
    # Model 4: Lorenz attractor's z-coordinate
    # --------------------------------------
    def generate_lorenz_z(self) -> np.ndarray:
        dt_sim = 0.01
        initial_state = np.array([1.0, -1.0, -1.0])
        _, trajectory = self._lorenz_dynamics([0, (self.num_steps - 1) * dt_sim], dt_sim, initial_state)
        z_values = trajectory[:, 2]  # third coordinate
        noisy_signal = z_values + np.sqrt(self.noise_variance) * self.rng.standard_normal(len(z_values))
        return noisy_signal

    def generate_lorenz_3d(
            self,
            initial_state: Tuple[float, float, float] = (-1.0, 1.0, -1.0),
            time_step: float = 0.01,
            number_of_channels=10,
            device='cpu'):
        """
        Generate 3D Lorenz system trajectories and noisy observations.

        This function simulates the Lorenz system using the given initial state and
        parameters, centers the latent states, and projects them into the observation
        space with Gaussian noise.

        Args:
            initial_state (Tuple[float, float, float], optional):
                Initial condition (x₀, y₀, z₀) for the Lorenz system.
                Defaults to (-1.0, 1.0, -1.0).
            time_step (float, optional):
                Time step for the uniform simulation grid. Defaults to 0.01.

        Returns:
            Dict
        """
        N = number_of_channels
        W = 0.1 * torch.randn(N, 3, device=device)

        diag_elems = self.noise_variance * torch.abs(torch.randn(N, device=device))
        z_var = torch.diag(diag_elems)

        # Step 1: Simulate Lorenz system on uniform grid
        t_start, t_end = 0.0, (self.num_steps - 1) * time_step
        time_uniform, states_uniform = self.simulate_lorenz(
            (t_start, t_end), time_step, initial_state
        )

        # Step 2: Convert to torch tensor and center
        latent_states = torch.from_numpy(states_uniform.T).float()  # [3, K]
        latent_states -= latent_states.mean(dim=1, keepdim=True)

        # Step 3: Project to observation space and add Gaussian noise
        cholesky_factor = torch.linalg.cholesky(z_var)  # [N, N]
        noise = cholesky_factor @ torch.randn(N, self.num_steps)
        observations = W @ latent_states + noise

        data = {
            'observation_z': observations,
            'latent': latent_states,
            't': time_uniform
        }

        return data

    def simulate_lorenz(
            self,
            time_span: Tuple[float, float],
            time_step: float,
            initial_state: Tuple[float, float, float],
            sigma: float = 10.0,
            beta: float = 8 / 3,
            rho: float = 14.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Lorenz system and interpolate results onto a uniform grid.

        Args:
            time_span (Tuple[float, float]):
                Start and end times for simulation.
            time_step (float):
                Desired time step for the uniform output grid.
            initial_state (Tuple[float, float, float]):
                Initial state (x₀, y₀, z₀) for the Lorenz system.
            sigma (float, optional):
                Lorenz σ parameter. Defaults to 10.0.
            beta (float, optional):
                Lorenz β parameter. Defaults to 8/3.
            rho (float, optional):
                Lorenz ρ parameter. Defaults to 14.0.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - `time_uniform` (shape [M,]): Uniformly spaced time points.
                - `states_uniform` (shape [M, 3]): Simulated Lorenz states [x, y, z]
                  at each time point.
        """

        def lorenz_ode(_, state):
            dx_dt = sigma * (state[1] - state[0])
            dy_dt = state[0] * (rho - state[2]) - state[1]
            dz_dt = state[0] * state[1] - beta * state[2]
            return [dx_dt, dy_dt, dz_dt]

        # Step 1: Integrate with adaptive solver
        solution = solve_ivp(
            lorenz_ode,
            (time_span[0], time_span[1]),
            initial_state,
            method='RK45'
        )

        # Step 2: Build uniform time grid
        time_uniform = np.arange(time_span[0], time_span[1] + time_step, time_step)
        time_uniform = time_uniform[time_uniform <= time_span[1]]

        # Step 3: Linear interpolation for each dimension
        states_uniform = np.empty((time_uniform.size, 3), dtype=float)
        for dim in range(3):
            states_uniform[:, dim] = np.interp(
                time_uniform, solution.t, solution.y[dim, :]
            )

        return time_uniform, states_uniform

    # --------------------------------------
    # Internal: Lorenz dynamics solver
    # --------------------------------------
    @staticmethod
    def _lorenz_dynamics(t_span, dt, initial_state):
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0

        def lorenz_rhs(state):
            x, y, z = state
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return np.array([dx, dy, dz])

        t0, tf = t_span
        t_vals = np.arange(t0, tf + dt, dt)
        y_vals = np.zeros((len(t_vals), 3))
        y_vals[0] = initial_state

        for i in range(len(t_vals) - 1):
            k1 = lorenz_rhs(y_vals[i])
            k2 = lorenz_rhs(y_vals[i] + dt * k1 / 2)
            k3 = lorenz_rhs(y_vals[i] + dt * k2 / 2)
            k4 = lorenz_rhs(y_vals[i] + dt * k3)
            y_vals[i + 1] = y_vals[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return t_vals, y_vals

    # --------------------------------------
    # Visualization Utilities
    # --------------------------------------
    def _plot_data(self, data: np.ndarray, title: str, xlabel: str = "Time step", ylabel: str = "Value",
                   show: bool = True, figsize=(8, 4)):
        """Generic plotting helper."""
        plt.figure(figsize=figsize)
        plt.plot(data, lw=1.5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if show:
            plt.show()

    def plot_exponentiated_cosine(self, show: bool = True):
        """Generate and plot exponentiated cosine wave."""
        data = self.generate_exponentiated_cosine()
        self._plot_data(data, "Exponentiated Cosine Wave", show=show)

    def plot_power_law(self, show: bool = True):
        """Generate and plot power-law growth."""
        data = self.generate_power_law()
        self._plot_data(data, "Power-Law Growth", show=show)

    def plot_reversed_chirp(self, show: bool = True):
        """Generate and plot reversed chirp."""
        data = self.generate_reversed_chirp()
        self._plot_data(data, "Reversed Chirp", show=show)

    def plot_lorenz_z(self, show: bool = True):
        """Generate and plot Lorenz attractor's z-coordinate."""
        data = self.generate_lorenz_z()
        self._plot_data(data, "Lorenz Attractor Z-Coordinate", show=show)

    def plot_lorenz_3d(
            self,
            latent_states: torch.Tensor | np.ndarray,
            *,
            color_by_time: bool = True,
            linewidth: float = 1.2,
            figsize: tuple[float, float] = (7.0, 5.5),
            elev: float = 25.0,
            azim: float = -60.0,
            save_path: str | None = None,
            show: bool = True,
    ) -> plt.Axes:
        """
        Plot a 3D Lorenz trajectory.

        Args:
            latent_states: Array/Tensor of shape (3, K) or (K, 3) containing [x, y, z].
            color_by_time: If True, apply a time gradient along the path. Defaults to True.
            linewidth: Line width for the trajectory. Defaults to 1.2.
            figsize: Size of the matplotlib figure. Defaults to (7.0, 5.5).
            elev: Elevation angle in the z plane for the 3D axes. Defaults to 25.0.
            azim: Azimuth angle in the x,y plane for the 3D axes. Defaults to -60.0.
            save_path: If provided, save the figure to this path (format inferred from suffix).
            show: If True, call plt.show() at the end. Defaults to True.

        Returns:
            The matplotlib Axes3D object for further customization.
        """
        # ----- Prepare data as (K, 3) numpy array -----
        if isinstance(latent_states, torch.Tensor):
            arr = latent_states.detach().to("cpu").numpy()
        else:
            arr = np.asarray(latent_states)

        if arr.shape[0] == 3 and arr.ndim == 2:
            arr = arr.T  # -> (K, 3)

        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("`latent_states` must have shape (3, K) or (K, 3).")

        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        k = arr.shape[0]

        # ----- Create figure/axes -----
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        # ----- Plot with optional time gradient -----
        if color_by_time and k > 1:
            # Build line segments so we can color by time along the path
            points = arr.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Normalize time 0..1 for colormap
            t = np.linspace(0.0, 1.0, k - 1)
            lc = Line3DCollection(segments, cmap="viridis", linewidth=linewidth)
            lc.set_array(t)
            ax.add_collection3d(lc)

            # Autoscale limits
            ax.auto_scale_xyz(x, y, z)

            # Colorbar
            cbar = fig.colorbar(lc, ax=ax, pad=0.1, fraction=0.03)
            cbar.set_label("Relative time")
        else:
            ax.plot(x, y, z, linewidth=linewidth)

        # Mark start (green) and end (red) points for orientation
        ax.scatter([x[0]], [y[0]], [z[0]], s=30, label="start")
        ax.scatter([x[-1]], [y[-1]], [z[-1]], s=30, label="end")

        # ----- Labels & aesthetics -----
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Lorenz Attractor (3D)")
        ax.legend(loc="upper left")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        return ax
