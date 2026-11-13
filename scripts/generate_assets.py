"""Generate high-resolution visual assets: GIFs and comparison plots.

Outputs generated into `assets/`:
1. `assets/nbody_orbits.gif`: Side-by-side animated visualization of Numerical vs Neural orbits.
2. `assets/energy_conservation.png`: Energy conservation comparison (Verlet vs Euler vs Neural).
3. `assets/rollout_comparison.png`: Trajectory overlay and error growth over rollout horizon.
4. `assets/runtime_benchmark.png`: Latency and throughput scaling across batch sizes.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from nbody.benchmark import benchmark_step_latencies
from nbody.dataset import sample_initial_conditions
from nbody.evaluate import evaluate_rollout
from nbody.model import NBodySurrogateMLP
from nbody.physics import compute_total_energy, compute_total_momentum
from nbody.simulator import EulerSimulator, VelocityVerletSimulator
from nbody.train import train_surrogate


def ensure_assets_dir() -> str:
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    return assets_dir


def generate_energy_plot(
    pos0: np.ndarray,
    vel0: np.ndarray,
    masses: np.ndarray,
    model: NBodySurrogateMLP,
    assets_dir: str,
    steps: int = 400,
    dt: float = 0.01,
):
    """Plot energy conservation across Verlet, Euler, and Neural Surrogate."""
    print("[Assets] Generating energy conservation plot...")
    sim_vv = VelocityVerletSimulator(masses=masses, dt=dt)
    vv_res = sim_vv.simulate(pos0, vel0, steps=steps, track_energy=True)

    sim_eu = EulerSimulator(masses=masses, dt=dt)
    eu_res = sim_eu.simulate(pos0, vel0, steps=steps)

    neural_pos, neural_vel = model.rollout(pos0, vel0, steps=steps)
    neural_e = np.zeros(steps + 1)
    neural_rel_err = np.zeros(steps + 1)
    e0 = vv_res.total_energies[0]

    for t in range(steps + 1):
        et, _, _ = compute_total_energy(neural_pos[t], neural_vel[t], masses)
        neural_e[t] = et
        neural_rel_err[t] = abs((et - e0) / (abs(e0) + 1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    plt.subplots_adjust(wspace=0.28)

    # 1. Energy Components (Verlet)
    ax1 = axes[0]
    ax1.plot(vv_res.times, vv_res.total_energies, label="Total Energy $E$", color="#2b5c8f", lw=2)
    ax1.plot(vv_res.times, vv_res.kinetic_energies, label="Kinetic $T$", color="#e05d44", lw=1.5, ls="--")
    ax1.plot(vv_res.times, vv_res.potential_energies, label="Potential $U$", color="#4caf50", lw=1.5, ls=":")
    ax1.set_title("Velocity Verlet Energy Partition", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time $t$", fontsize=11)
    ax1.set_ylabel("Energy", fontsize=11)
    ax1.legend(loc="center right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. Relative Energy Drift (Log Scale)
    ax2 = axes[1]
    ax2.semilogy(vv_res.times, np.maximum(vv_res.relative_energy_errors, 1e-14),
                 label="Velocity Verlet (Symplectic)", color="#1976d2", lw=2)
    ax2.semilogy(vv_res.times, np.maximum(neural_rel_err, 1e-14),
                 label="Neural Surrogate (ResMLP)", color="#ff9800", lw=2)
    ax2.semilogy(eu_res.times, np.maximum(eu_res.relative_energy_errors, 1e-14),
                 label="Forward Euler (1st Order)", color="#d32f2f", lw=1.8, ls="--")
    ax2.set_title("Relative Energy Error $|(E(t) - E_0) / E_0|$", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("Relative Error (log scale)", fontsize=11)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, which="both", alpha=0.3)

    # 3. Momentum Conservation Drift
    ax3 = axes[2]
    p_vv = [np.linalg.norm(compute_total_momentum(v, masses)) for v in vv_res.velocities]
    p_nn = [np.linalg.norm(compute_total_momentum(v, masses)) for v in neural_vel]
    ax3.plot(vv_res.times, p_vv, label="Velocity Verlet", color="#1976d2", lw=2)
    ax3.plot(vv_res.times, p_nn, label="Neural Surrogate", color="#ff9800", lw=2)
    ax3.set_title(r"Total Linear Momentum Magnitude $||\mathbf{P}(t)||$", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Time $t$", fontsize=11)
    ax3.set_ylabel("Momentum Magnitude", fontsize=11)
    ax3.legend(loc="upper left", frameon=True)
    ax3.grid(True, alpha=0.3)

    out_path = os.path.join(assets_dir, "energy_conservation.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Assets] Saved {out_path}")


def generate_rollout_comparison_plot(
    eval_res,
    assets_dir: str,
):
    """Plot spatial trajectories and error accumulation over horizon."""
    print("[Assets] Generating trajectory & rollout comparison plot...")
    gt_p = eval_res.gt_positions
    nn_p = eval_res.neural_positions
    times = eval_res.times
    n_bodies = gt_p.shape[1]

    colors = ["#2196f3", "#e91e63", "#4caf50", "#9c27b0", "#ff9800"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. 2D Trajectories Overlay
    ax1 = axes[0]
    for i in range(n_bodies):
        c = colors[i % len(colors)]
        ax1.plot(gt_p[:, i, 0], gt_p[:, i, 1], color=c, lw=2.2, label=f"Body {i+1} (GT Verlet)")
        ax1.plot(nn_p[:, i, 0], nn_p[:, i, 1], color=c, lw=1.8, ls="--", alpha=0.85, label=f"Body {i+1} (Neural)")
        ax1.scatter([gt_p[0, i, 0]], [gt_p[0, i, 1]], color=c, s=60, marker="o", edgecolors="black", zorder=5)

    ax1.set_title("2D Orbits: Ground Truth vs. Neural Surrogate", fontsize=13, fontweight="bold")
    ax1.set_xlabel("$x$", fontsize=11)
    ax1.set_ylabel("$y$", fontsize=11)
    ax1.axis("equal")
    ax1.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. Per-Step Rollout MSE Growth
    ax2 = axes[1]
    ax2.plot(times, eval_res.pos_mse_per_step, color="#d32f2f", lw=2, label="Position MSE $\\frac{1}{N}\\sum ||x_{pred} - x_{gt}||^2$")
    ax2.plot(times, eval_res.vel_mse_per_step, color="#1976d2", lw=2, ls="--", label="Velocity MSE $\\frac{1}{N}\\sum ||v_{pred} - v_{gt}||^2$")
    ax2.set_title("Autoregressive Rollout Error Accumulation", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Time $t$", fontsize=11)
    ax2.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, alpha=0.3)

    out_path = os.path.join(assets_dir, "rollout_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Assets] Saved {out_path}")


def generate_benchmark_plot(
    model: NBodySurrogateMLP,
    assets_dir: str,
    device: torch.device,
):
    """Plot latency and throughput curves across batch sizes."""
    print("[Assets] Generating runtime benchmark plot...")
    batch_sizes = [1, 4, 16, 64, 256, 1024]
    latencies = benchmark_step_latencies(model, batch_sizes=batch_sizes, device=device)

    v_times = [latencies["numerical_verlet"][b] for b in batch_sizes]
    n_times = [latencies["neural_surrogate"][b] for b in batch_sizes]

    v_thru = [b / (latencies["numerical_verlet"][b] / 1000.0) for b in batch_sizes]
    n_thru = [b / (latencies["neural_surrogate"][b] / 1000.0) for b in batch_sizes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Step Latency
    ax1 = axes[0]
    ax1.plot(batch_sizes, v_times, marker="o", lw=2, color="#1976d2", label="Numerical Verlet (NumPy)")
    ax1.plot(batch_sizes, n_times, marker="s", lw=2, color="#e05d44", label=f"Neural Surrogate ({device.type.upper()})")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Single-Step Latency vs. Batch Size", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Batch Size $B$", fontsize=11)
    ax1.set_ylabel("Step Latency (ms)", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, which="both", alpha=0.3)

    # 2. Throughput
    ax2 = axes[1]
    ax2.plot(batch_sizes, v_thru, marker="o", lw=2, color="#1976d2", label="Numerical Verlet (NumPy)")
    ax2.plot(batch_sizes, n_thru, marker="s", lw=2, color="#e05d44", label=f"Neural Surrogate ({device.type.upper()})")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title("Simulation Throughput vs. Batch Size", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Batch Size $B$", fontsize=11)
    ax2.set_ylabel("States Evaluated / Second", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, which="both", alpha=0.3)

    out_path = os.path.join(assets_dir, "runtime_benchmark.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Assets] Saved {out_path}")


def generate_animation_gif(
    gt_positions: np.ndarray,
    neural_positions: np.ndarray,
    assets_dir: str,
    fps: int = 25,
    trail_len: int = 40,
    stride: int = 2,
):
    """Render side-by-side animated GIF of numerical vs neural trajectories."""
    print("[Assets] Rendering trajectory animated GIF (assets/nbody_orbits.gif)...")
    gt_p = gt_positions[::stride]
    nn_p = neural_positions[::stride]
    T, N, _ = gt_p.shape

    # Determine unified axis bounds with padding
    all_x = np.concatenate([gt_p[..., 0].flatten(), nn_p[..., 0].flatten()])
    all_y = np.concatenate([gt_p[..., 1].flatten(), nn_p[..., 1].flatten()])
    margin = 0.3
    x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
    y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin

    fig, (ax_gt, ax_nn) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    fig.patch.set_facecolor("#0e1117")

    colors = ["#00e5ff", "#ff007f", "#ffee55", "#76ff03"]

    for ax, title in [(ax_gt, "Numerical Ground Truth (Velocity Verlet)"),
                     (ax_nn, "Neural Surrogate Rollout (PyTorch ResMLP)")]:
        ax.set_facecolor("#161b22")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, color="#30363d", alpha=0.5, linestyle="--")

    # Lines and scatter elements
    lines_gt = [ax_gt.plot([], [], color=colors[i % len(colors)], lw=2, alpha=0.8)[0] for i in range(N)]
    scat_gt = [ax_gt.plot([], [], marker="o", color=colors[i % len(colors)], markersize=8,
                           markeredgecolor="white", markeredgewidth=1.2)[0] for i in range(N)]

    lines_nn = [ax_nn.plot([], [], color=colors[i % len(colors)], lw=2, alpha=0.8)[0] for i in range(N)]
    scat_nn = [ax_nn.plot([], [], marker="o", color=colors[i % len(colors)], markersize=8,
                           markeredgecolor="white", markeredgewidth=1.2)[0] for i in range(N)]

    time_text = fig.text(0.5, 0.03, "", ha="center", color="white", fontsize=11, fontweight="medium")

    def init():
        for line, scat in zip(lines_gt + lines_nn, scat_gt + scat_nn):
            line.set_data([], [])
            scat.set_data([], [])
        time_text.set_text("")
        return lines_gt + lines_nn + scat_gt + scat_nn + [time_text]

    def update(frame):
        t_start = max(0, frame - trail_len)
        for i in range(N):
            # GT
            lines_gt[i].set_data(gt_p[t_start : frame + 1, i, 0], gt_p[t_start : frame + 1, i, 1])
            scat_gt[i].set_data([gt_p[frame, i, 0]], [gt_p[frame, i, 1]])

            # Neural
            lines_nn[i].set_data(nn_p[t_start : frame + 1, i, 0], nn_p[t_start : frame + 1, i, 1])
            scat_nn[i].set_data([nn_p[frame, i, 0]], [nn_p[frame, i, 1]])

        time_text.set_text(f"Simulation Step: {frame * stride} / {T * stride}")
        return lines_gt + lines_nn + scat_gt + scat_nn + [time_text]

    anim = animation.FuncAnimation(
        fig, update, frames=T, init_func=init, blit=True, interval=1000 // fps
    )

    out_gif = os.path.join(assets_dir, "nbody_orbits.gif")
    anim.save(out_gif, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"[Assets] Saved {out_gif}")


def main():
    assets_dir = ensure_assets_dir()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Assets] Using device: {device}")

    # Train or load surrogate model
    ckpt_path = "checkpoints/best_surrogate.pt"
    if os.path.exists(ckpt_path):
        print(f"[Assets] Loading existing surrogate from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = NBodySurrogateMLP(
            n_bodies=ckpt.get("n_bodies", 3),
            hidden_dim=ckpt.get("hidden_dim", 256),
            n_layers=ckpt.get("n_layers", 4),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("stats") is not None:
            model.set_normalization(ckpt["stats"])
    else:
        print("[Assets] Training high-quality surrogate model for asset generation...")
        model, _ = train_surrogate(
            n_bodies=3,
            n_train_trajectories=120,
            n_val_trajectories=30,
            steps_per_trajectory=250,
            epochs=35,
            hidden_dim=256,
            n_layers=4,
            save_path=ckpt_path,
            device=device.type,
            seed=42,
        )

    # Evaluation on a figure-8 test trajectory
    pos0, vel0, masses = sample_initial_conditions(n_bodies=3, scenario="figure_eight", seed=101)
    eval_res = evaluate_rollout(
        model=model,
        pos0=pos0,
        vel0=vel0,
        masses=masses,
        steps=250,
        dt=0.01,
        device=device,
    )

    generate_energy_plot(pos0, vel0, masses, model, assets_dir, steps=300)
    generate_rollout_comparison_plot(eval_res, assets_dir)
    generate_benchmark_plot(model, assets_dir, device)
    generate_animation_gif(eval_res.gt_positions, eval_res.neural_positions, assets_dir, fps=25, trail_len=35)
    print("\n[Assets] All visual assets generated successfully in assets/!")


if __name__ == "__main__":
    main()
