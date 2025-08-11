import torch
import torch.distributions as D
from matplotlib import pyplot as plt

from src.dataset import SyntheticDataGenerator
from src.models.model import HierarchicalSDEParticleFilter
from src.utils.utils import extract_curves
from src.utils.visualization_utils import plot_training_curves, plot_inference_and_inducing_points

# Parameters from paper
mode = '3d'
duration_sec = 250  # signal duration
sampling_rate_hz = 2  # samples per second
num_samples = int(duration_sec * sampling_rate_hz)  # 500
time_step = 1 / sampling_rate_hz  # 0.5 sec
noise_variance = 0.1
random_seed = 42  # For reproducibility

# Create generator
gen = SyntheticDataGenerator(
    num_steps=num_samples,
    time_step=time_step,
    noise_variance=noise_variance,
    random_seed=random_seed
)

# Generate chirp signal
if mode == '1d':
    synthetic_data = gen.generate_reversed_chirp()

    t = synthetic_data["t"]  # (500,)
    observation_z = synthetic_data["observation_z"]  # (500, 1) noisy observed signal
    latent = synthetic_data["latent"]  # (500,)
    # Plot result
    gen._plot_data(
        observation_z,
        title="Chirp Signal with Additive Gaussian Noise (σ² = 0.1)",
        xlabel="Sample Index",
        ylabel="Amplitude",
        figsize=(12, 5)
    )

else:
    synthetic_data = gen.generate_lorenz_3d()
    t = synthetic_data["t"]  # (500,)
    observation_z = synthetic_data["observation_z"]  # (500, 1) noisy observed signal
    latent = synthetic_data["latent"]  # (500,)

    gen.plot_lorenz_3d(latent)


