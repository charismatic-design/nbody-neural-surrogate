"""N-Body Neural Surrogate Package."""

from nbody.benchmark import benchmark_rollout_horizon, benchmark_step_latencies
from nbody.dataset import NBodyDataset, generate_trajectory_dataset, sample_initial_conditions
from nbody.evaluate import RolloutEvaluation, evaluate_rollout
from nbody.model import NBodySurrogateMLP
from nbody.physics import (
    compute_center_of_mass,
    compute_gravitational_accelerations,
    compute_kinetic_energy,
    compute_potential_energy,
    compute_total_energy,
    compute_total_momentum,
)
from nbody.simulator import (
    EulerSimulator,
    SciPySimulator,
    SimulationResult,
    VelocityVerletSimulator,
)
from nbody.train import train_surrogate

__all__ = [
    "compute_gravitational_accelerations",
    "compute_kinetic_energy",
    "compute_potential_energy",
    "compute_total_energy",
    "compute_center_of_mass",
    "compute_total_momentum",
    "VelocityVerletSimulator",
    "EulerSimulator",
    "SciPySimulator",
    "SimulationResult",
    "sample_initial_conditions",
    "generate_trajectory_dataset",
    "NBodyDataset",
    "NBodySurrogateMLP",
    "train_surrogate",
    "evaluate_rollout",
    "RolloutEvaluation",
    "benchmark_step_latencies",
    "benchmark_rollout_horizon",
]
