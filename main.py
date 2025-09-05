from src.dataset import SyntheticDataGenerator
from src.models.hsde.hsde_3d_model import HiSDE3D, plot_hisde_3d
from src.utils.visualization_utils import plot_hisde_3d_matplotlib

# Parameters from paper
device = 'cuda'
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
    random_seed=random_seed,
    device=device
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
    observation_z = synthetic_data["observation_z"]  # (500, 10) noisy observed signal
    latent = synthetic_data["latent"]  # (500, 3)

    gen.plot_lorenz_3d(latent)



# 1) Initialize 3D HiSDE model
model = HiSDE3D(device=device)

# 2) Fit the model
data = model.fit(observation_z, in_loop=2, num_iters=2)

# 4) Extract final trajectories & marks
Xs = data['Xs']  # [3, sample, K+1]
Ys = data['Ys']  # [3, sample, K+1]
Ms = data['Ms']  # [3, sample, max_events]   <— NEW

# 5) Compute inducing‐point indices for plotting
raw_tau = data['Ts'][0, : model.event_cnt]
tau_idx = (raw_tau / model.dt).round().long().cpu().numpy().astype(int)

# 6) Drop t=0 and convert to numpy
T = observation_z.shape[1]
Ys_plot = Ys[:, :, 1:T + 1]
Xs_plot = Xs[:, :, 1:T + 1]

Z_np = observation_z.cpu().numpy()
Ys_np = Ys_plot.cpu().numpy()
Y_true_np = latent.cpu().numpy()
Xs_np = Xs_plot.cpu().numpy()
Ms_np = Ms.cpu().numpy()  # <— NEW

plot_hisde_3d_matplotlib(Z=Z_np,
                         Y_fit=Ys_plot,
                         Y_true=Y_true_np,
                         Xs=Xs_np,
                         Ms=Ms_np,
                         dt=model.dt,
                         tau_idx=tau_idx)

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
