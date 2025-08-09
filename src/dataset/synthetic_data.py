import numpy as np
import matplotlib.pyplot as plt
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
        noisy_signal = transformed + np.sqrt(self.noise_variance) * self.rng.standard_normal(len(signal))
        zero_mean_signal = noisy_signal - np.mean(noisy_signal)
        return zero_mean_signal

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
            y_vals[i + 1] = y_vals[i] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return t_vals, y_vals

    # --------------------------------------
    # Visualization Utilities
    # --------------------------------------
    def _plot_data(self, data: np.ndarray, title: str, xlabel: str = "Time step", ylabel: str = "Value", show: bool = True):
        """Generic plotting helper."""
        plt.figure(figsize=(8, 4))
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
