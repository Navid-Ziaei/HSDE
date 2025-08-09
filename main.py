from src.dataset import SyntheticDataGenerator

# Create reproducible generator
gen = SyntheticDataGenerator(num_steps=500, time_step=0.05, noise_variance=0.1, random_seed=42)

# Plot each model
gen.plot_exponentiated_cosine()
gen.plot_power_law()
gen.plot_reversed_chirp()
gen.plot_lorenz_z()
