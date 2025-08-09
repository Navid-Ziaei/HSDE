import torch
import torch.distributions as D
from matplotlib import pyplot as plt

from src.dataset import SyntheticDataGenerator
from src.models.model import HierarchicalSDEParticleFilter
from src.utils.utils import extract_curves
from src.utils.visualization_utils import plot_training_curves, plot_inference_and_inducing_points

# Parameters from paper
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
synthetic_dat = gen.generate_reversed_chirp()

t = synthetic_dat["t"]  # (500,)
chirp_signal = synthetic_dat["Z"]  # (500, 1) noisy observed signal
clean = synthetic_dat["clean"]  # (500,)
# Plot result
gen._plot_data(
    chirp_signal,
    title="Chirp Signal with Additive Gaussian Noise (σ² = 0.1)",
    xlabel="Sample Index",
    ylabel="Amplitude",
    figsize=(12, 5)
)

# Convert to torch tensor for SMC
Z = torch.tensor(chirp_signal, dtype=torch.float32).unsqueeze(1)  # shape (K, 1)

# ==== 5. Create particle filter ====

# --- Model (Step 2 structure + Step 3 init) ---
M = 1  # observed dimension (our chirp is 1-D)
D = 1  # latent dimension (start with 1 to keep simple)
config = {
    "num_particles": 10000,
    "em_iters": 20
}
model = HierarchicalSDEParticleFilter(M=M,
                                      D_latent=D,
                                      dt=time_step,
                                      num_steps=num_samples,
                                      sigma_x=1e-1,  # [Sec. \ref{a33}]
                                      sigma_y=1e-4,
                                      k_tau=20.0,
                                      lmb_tau=0.5,
                                      config=config)

# Apply "paper" settings:
model.apply_paper_initialization(
    obs_noise_variance=noise_variance,  # known R [Eq. \ref{eq6}]
    mean_wait_time_sec=40.0,  # Gamma mean [Eq. (2)]
    std_wait_time_sec=8.94,  # Gamma std  [Eq. (2)]
    sigma_x=1e-1,  # [Sec. \ref{a33}]
    sigma_y=1e-4,  # [Sec. \ref{a33}]
    num_particles=10000,  # [Alg. \ref{alg1}, \ref{alg2}]
    em_iters=20,  # [Sec. \ref{inference}, \ref{a43}]
    fix_sigma_y=True  # per note in [\ref{a43}]
)

print("Gamma k (shape):", float(model.k_tau))
print("Gamma λ (rate): ", float(model.lmb_tau))
print("Gamma mean (k/λ) ≈ 40:", float(model.k_tau / model.lmb_tau))
print("Gamma std  (sqrt(k)/λ) ≈ 8.94:",
      float(torch.sqrt(model.k_tau) / model.lmb_tau))
print("sigma_x:", float(model.sigma_x))
print("sigma_y:", float(model.sigma_y))
print("#particles:", model.config["num_particles"], "| EM iters:", model.config["em_iters"])
print("R diag:", model.R_diag.detach().cpu().numpy())

train_history = model.fit(Z,
                          max_iter=5,
                          seed=123,
                          eval_every=1,
                          eval_kwargs={"num_particles": 5000, "ess_fraction": 0.5, "seed": 7},
                          clean=torch.Tensor(clean),
                          save_best_path="results/hsde_best.pt"
                          )
print(train_history["params"])



curves = extract_curves(train_history)

# After training:
fig1 = plot_training_curves(curves)

# Trajectories with inducing points (and optional clean target)
fig2 = plot_inference_and_inducing_points(
    model,
    Z,
    clean=torch.Tensor(clean),
    num_particles=5000,
    ess_fraction=0.5,
    seed=7,
)
plt.show()

# Save final (with history metadata)
model.save("results/hsde_final.pt", extra={"train_history": train_history})

# Step 5: inference + evaluation
#res = model.evaluate(Z, clean=torch.Tensor(clean), num_particles=5000, ess_fraction=0.5, seed=7)
print(res["metrics"])
