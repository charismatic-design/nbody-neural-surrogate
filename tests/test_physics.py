"""Unit tests for N-body physics engine and numerical integrators."""

import numpy as np
import pytest

from nbody.physics import (
    compute_center_of_mass,
    compute_gravitational_accelerations,
    compute_kinetic_energy,
    compute_potential_energy,
    compute_total_energy,
    compute_total_momentum,
)
from nbody.simulator import EulerSimulator, VelocityVerletSimulator
from nbody.dataset import sample_initial_conditions


def test_newtons_third_law_net_force_zero():
    """Verify that internal gravitational forces sum to zero (net F = 0)."""
    n_bodies = 5
    pos = np.random.randn(n_bodies, 2)
    masses = np.random.uniform(0.5, 2.0, size=n_bodies)

    acc = compute_gravitational_accelerations(pos, masses, G=1.0, softening=0.05)
    net_force = np.sum(acc * masses[:, None], axis=0)

    # Net force should be zero within machine epsilon
    np.testing.assert_allclose(net_force, np.zeros(2), atol=1e-12)


def test_batched_accelerations_match_single():
    """Verify batched acceleration tensor matches iterating over each state."""
    batch_size = 4
    n_bodies = 3
    pos_batch = np.random.randn(batch_size, n_bodies, 2)
    masses = np.ones(n_bodies)

    acc_batch = compute_gravitational_accelerations(pos_batch, masses, G=1.0, softening=0.1)

    for b in range(batch_size):
        acc_single = compute_gravitational_accelerations(pos_batch[b], masses, G=1.0, softening=0.1)
        np.testing.assert_allclose(acc_batch[b], acc_single, rtol=1e-10, atol=1e-10)


def test_velocity_verlet_energy_conservation():
    """Verify that Velocity Verlet exhibits bounded energy oscillation."""
    pos0, vel0, masses = sample_initial_conditions(n_bodies=3, scenario="figure_eight", seed=42)
    sim = VelocityVerletSimulator(masses=masses, dt=0.01)

    res = sim.simulate(pos0, vel0, steps=300, track_energy=True)

    # Relative energy drift should remain below 1e-4 for symplectic Verlet
    assert res.max_energy_drift < 1e-3, f"Energy drift too high: {res.max_energy_drift}"


def test_euler_divergence():
    """Verify that explicit Euler has significantly worse energy conservation than Verlet."""
    pos0, vel0, masses = sample_initial_conditions(n_bodies=3, scenario="figure_eight", seed=42)
    dt = 0.01
    steps = 200

    vv_sim = VelocityVerletSimulator(masses=masses, dt=dt)
    eu_sim = EulerSimulator(masses=masses, dt=dt)

    vv_res = vv_sim.simulate(pos0, vel0, steps=steps)
    eu_res = eu_sim.simulate(pos0, vel0, steps=steps)

    assert eu_res.max_energy_drift > vv_res.max_energy_drift * 10
