"""Evaluation and comparison metrics between Numerical Verlet and Neural Surrogate."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from nbody.model import NBodySurrogateMLP
from nbody.physics import compute_total_energy
from nbody.simulator import EulerSimulator, VelocityVerletSimulator


@dataclass
class RolloutEvaluation:
    """Evaluation metrics comparing numerical ground truth vs neural surrogate."""

    horizon_steps: int
    times: np.ndarray
    gt_positions: np.ndarray      # (T, N, 2)
    gt_velocities: np.ndarray     # (T, N, 2)
    neural_positions: np.ndarray  # (T, N, 2)
    neural_velocities: np.ndarray # (T, N, 2)
    euler_positions: Optional[np.ndarray] = None
    euler_velocities: Optional[np.ndarray] = None

    pos_mse_per_step: Optional[np.ndarray] = None     # (T,)
    vel_mse_per_step: Optional[np.ndarray] = None     # (T,)
    gt_energy: Optional[np.ndarray] = None            # (T,)
    neural_energy: Optional[np.ndarray] = None        # (T,)
    euler_energy: Optional[np.ndarray] = None         # (T,)
    gt_energy_rel_err: Optional[np.ndarray] = None    # (T,)
    neural_energy_rel_err: Optional[np.ndarray] = None# (T,)
    euler_energy_rel_err: Optional[np.ndarray] = None # (T,)

    @property
    def cumulative_pos_mse(self) -> float:
        return float(np.mean(self.pos_mse_per_step))

    @property
    def cumulative_vel_mse(self) -> float:
        return float(np.mean(self.vel_mse_per_step))


def evaluate_rollout(
    model: NBodySurrogateMLP,
    pos0: np.ndarray,
    vel0: np.ndarray,
    masses: np.ndarray,
    steps: int = 150,
    dt: float = 0.01,
    G: float = 1.0,
    softening: float = 0.05,
    include_euler: bool = True,
    device: Optional[torch.device] = None,
) -> RolloutEvaluation:
    """Compare Velocity Verlet numerical simulation against neural surrogate rollout.

    Parameters
    ----------
    model : NBodySurrogateMLP
    pos0 : np.ndarray, shape (N, 2)
    vel0 : np.ndarray, shape (N, 2)
    masses : np.ndarray, shape (N,)
    steps : int
    dt : float
    G : float
    softening : float
    include_euler : bool
    device : torch.device, optional

    Returns
    -------
    RolloutEvaluation
    """
    sim_vv = VelocityVerletSimulator(masses=masses, G=G, softening=softening, dt=dt)
    gt_res = sim_vv.simulate(pos0, vel0, steps=steps, track_energy=True)

    neural_pos, neural_vel = model.rollout(pos0, vel0, steps=steps, device=device)

    # Calculate step-by-step errors
    pos_err = np.mean((neural_pos - gt_res.positions) ** 2, axis=(1, 2))
    vel_err = np.mean((neural_vel - gt_res.velocities) ** 2, axis=(1, 2))

    # Calculate neural rollout energy
    neural_energies = np.zeros(steps + 1, dtype=np.float64)
    neural_rel_err = np.zeros(steps + 1, dtype=np.float64)
    e0 = gt_res.total_energies[0]

    for t in range(steps + 1):
        et, _, _ = compute_total_energy(
            neural_pos[t], neural_vel[t], masses, G=G, softening=softening
        )
        neural_energies[t] = et
        neural_rel_err[t] = abs((et - e0) / (abs(e0) + 1e-12))

    euler_pos, euler_vel, euler_e, euler_rel = None, None, None, None
    if include_euler:
        sim_eu = EulerSimulator(masses=masses, G=G, softening=softening, dt=dt)
        eu_res = sim_eu.simulate(pos0, vel0, steps=steps)
        euler_pos = eu_res.positions
        euler_vel = eu_res.velocities
        euler_e = eu_res.total_energies
        euler_rel = eu_res.relative_energy_errors

    return RolloutEvaluation(
        horizon_steps=steps,
        times=gt_res.times,
        gt_positions=gt_res.positions,
        gt_velocities=gt_res.velocities,
        neural_positions=neural_pos,
        neural_velocities=neural_vel,
        euler_positions=euler_pos,
        euler_velocities=euler_vel,
        pos_mse_per_step=pos_err,
        vel_mse_per_step=vel_err,
        gt_energy=gt_res.total_energies,
        neural_energy=neural_energies,
        euler_energy=euler_e,
        gt_energy_rel_err=gt_res.relative_energy_errors,
        neural_energy_rel_err=neural_rel_err,
        euler_energy_rel_err=euler_rel,
    )
