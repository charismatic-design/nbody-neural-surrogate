"""Dataset generation and PyTorch Dataset classes for N-body dynamical systems."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from nbody.physics import compute_center_of_mass, compute_total_momentum
from nbody.simulator import VelocityVerletSimulator


def sample_initial_conditions(
    n_bodies: int = 3,
    scenario: str = "random_bounded",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate physically plausible initial conditions with zero total linear momentum.

    Parameters
    ----------
    n_bodies : int
        Number of bodies. Default 3.
    scenario : str
        'random_bounded', 'figure_eight', or 'central_mass'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    (positions, velocities, masses)
        positions: (N, 2)
        velocities: (N, 2)
        masses: (N,)
    """
    rng = np.random.default_rng(seed)

    if scenario == "figure_eight" and n_bodies == 3:
        # Famous Chenciner-Montgomery (2000) choreographic figure-8 orbit
        masses = np.ones(3, dtype=np.float64)
        x1, y1 = -0.97000436, 0.24308753
        vx3, vy3 = -2.0 * 0.46620531, -2.0 * 0.43236573
        vx1, vy1 = -0.5 * vx3, -0.5 * vy3

        positions = np.array([
            [x1, y1],
            [-x1, -y1],
            [0.0, 0.0],
        ], dtype=np.float64)
        velocities = np.array([
            [vx1, vy1],
            [vx1, vy1],
            [vx3, vy3],
        ], dtype=np.float64)

        # Add small perturbation if randomized
        if seed is not None:
            perturb_p = rng.normal(0, 0.02, size=positions.shape)
            perturb_v = rng.normal(0, 0.01, size=velocities.shape)
            positions += perturb_p
            velocities += perturb_v

    elif scenario == "central_mass":
        # 1 heavy central body, (n_bodies - 1) lighter orbiting bodies
        masses = np.ones(n_bodies, dtype=np.float64)
        masses[0] = 5.0  # heavy anchor
        positions = np.zeros((n_bodies, 2), dtype=np.float64)
        velocities = np.zeros((n_bodies, 2), dtype=np.float64)

        for i in range(1, n_bodies):
            radius = rng.uniform(0.8, 2.5)
            theta = rng.uniform(0, 2 * np.pi)
            positions[i] = [radius * np.cos(theta), radius * np.sin(theta)]

            # Approximate circular Keplerian orbital velocity v = sqrt(G*M/r)
            v_mag = np.sqrt(1.0 * masses[0] / radius) * rng.uniform(0.9, 1.1)
            # Tangent velocity direction
            velocities[i] = [-v_mag * np.sin(theta), v_mag * np.cos(theta)]

    else:  # 'random_bounded'
        # Equal or slightly varied masses
        masses = rng.uniform(0.8, 1.2, size=n_bodies)

        # Uniform disc distribution
        r = np.sqrt(rng.uniform(0.1, 1.5, size=n_bodies))
        theta = rng.uniform(0, 2 * np.pi, size=n_bodies)
        positions = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

        # Random velocities with bounded magnitude
        v_speed = rng.uniform(0.2, 0.7, size=n_bodies)
        v_theta = rng.uniform(0, 2 * np.pi, size=n_bodies)
        velocities = np.column_stack([v_speed * np.cos(v_theta), v_speed * np.sin(v_theta)])

    # Zero the center of mass and net linear momentum so system doesn't drift
    cm = compute_center_of_mass(positions, masses)
    positions -= cm

    total_p = compute_total_momentum(velocities, masses)
    velocities -= total_p / np.sum(masses)

    return positions, velocities, masses


def generate_trajectory_dataset(
    n_trajectories: int = 50,
    steps_per_trajectory: int = 200,
    n_bodies: int = 3,
    dt: float = 0.01,
    G: float = 1.0,
    softening: float = 0.05,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate multiple N-body trajectories using Velocity Verlet simulator.

    Returns dictionary containing:
        - positions: (n_trajectories, steps + 1, n_bodies, 2)
        - velocities: (n_trajectories, steps + 1, n_bodies, 2)
        - masses: (n_bodies,)
        - dt, G, softening
    """
    rng = np.random.default_rng(seed)
    masses = np.ones(n_bodies, dtype=np.float64)
    sim = VelocityVerletSimulator(masses=masses, G=G, softening=softening, dt=dt)

    pos_all = []
    vel_all = []

    for i in range(n_trajectories):
        traj_seed = seed + i * 1000 + 7
        # Mix scenarios: figure_eight perturbations and random bounded
        if i % 3 == 0 and n_bodies == 3:
            p0, v0, _ = sample_initial_conditions(n_bodies, "figure_eight", seed=traj_seed)
        elif i % 3 == 1:
            p0, v0, _ = sample_initial_conditions(n_bodies, "central_mass", seed=traj_seed)
        else:
            p0, v0, _ = sample_initial_conditions(n_bodies, "random_bounded", seed=traj_seed)

        res = sim.simulate(p0, v0, steps=steps_per_trajectory, track_energy=False)
        pos_all.append(res.positions)
        vel_all.append(res.velocities)

    return {
        "positions": np.array(pos_all, dtype=np.float32),   # (B, T, N, 2)
        "velocities": np.array(vel_all, dtype=np.float32),  # (B, T, N, 2)
        "masses": masses.astype(np.float32),
        "dt": dt,
        "G": G,
        "softening": softening,
    }


class NBodyDataset(Dataset):
    """PyTorch Dataset yielding single-step transition pairs (s_t -> delta_s_t).

    State s_t is concatenated positions and velocities of shape (4 * N,).
    Target delta_s_t is s_{t+1} - s_t of shape (4 * N,).
    """

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        normalize: bool = False,
        stats: Optional[Dict[str, float]] = None,
    ):
        """Parameters
        ----------
        positions : np.ndarray, shape (num_trajectories, T, N, 2)
        velocities : np.ndarray, shape (num_trajectories, T, N, 2)
        normalize : bool
        stats : dict, optional
        """
        B, T, N, _ = positions.shape
        self.n_bodies = N

        # Concatenate positions and velocities along last axis: (B, T, N, 4)
        states = np.concatenate([positions, velocities], axis=-1)  # (B, T, N, 4)
        states_flat = states.reshape(B, T, 4 * N)                  # (B, T, 4N)

        # Transition pairs
        inputs = states_flat[:, :-1, :].reshape(-1, 4 * N)
        next_states = states_flat[:, 1:, :].reshape(-1, 4 * N)
        targets = next_states - inputs  # Residual formulation!

        self.normalize = normalize
        if normalize:
            if stats is None:
                self.input_mean = inputs.mean(axis=0)
                self.input_std = inputs.std(axis=0) + 1e-7
                self.target_mean = targets.mean(axis=0)
                self.target_std = targets.std(axis=0) + 1e-7
            else:
                self.input_mean = stats["input_mean"]
                self.input_std = stats["input_std"]
                self.target_mean = stats["target_mean"]
                self.target_std = stats["target_std"]

            inputs = (inputs - self.input_mean) / self.input_std
            targets = (targets - self.target_mean) / self.target_std
        else:
            self.input_mean = np.zeros(4 * N, dtype=np.float32)
            self.input_std = np.ones(4 * N, dtype=np.float32)
            self.target_mean = np.zeros(4 * N, dtype=np.float32)
            self.target_std = np.ones(4 * N, dtype=np.float32)

        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def get_stats(self) -> Dict[str, np.ndarray]:
        return {
            "input_mean": self.input_mean,
            "input_std": self.input_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]
