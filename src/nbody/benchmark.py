"""Runtime and latency benchmarking between Velocity Verlet and Neural Surrogate."""

import time
from typing import Dict, List, Tuple
import numpy as np
import torch

from nbody.model import NBodySurrogateMLP
from nbody.physics import compute_gravitational_accelerations
from nbody.simulator import VelocityVerletSimulator


def benchmark_step_latencies(
    model: NBodySurrogateMLP,
    n_bodies: int = 3,
    batch_sizes: List[int] = [1, 16, 64, 256, 1024],
    n_repeats: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[int, float]]:
    """Measure single-step latency across varying batch sizes.

    Returns dict mapping:
      'numerical_verlet': {batch_size: latency_ms}
      'neural_surrogate': {batch_size: latency_ms}
    """
    if device is None:
        device = next(model.parameters()).device

    masses = np.ones(n_bodies, dtype=np.float64)
    sim = VelocityVerletSimulator(masses=masses)
    model.eval()

    results = {
        "numerical_verlet": {},
        "neural_surrogate": {},
    }

    # Warmup
    dummy_pos = np.random.randn(16, n_bodies, 2)
    dummy_vel = np.random.randn(16, n_bodies, 2)
    for _ in range(20):
        sim.step(dummy_pos, dummy_vel)
        with torch.no_grad():
            s = torch.randn(16, 4 * n_bodies, device=device)
            _ = model.step(s)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

    for B in batch_sizes:
        pos_b = np.random.randn(B, n_bodies, 2)
        vel_b = np.random.randn(B, n_bodies, 2)
        acc_b = compute_gravitational_accelerations(pos_b, masses)

        # 1. Numerical Verlet timing
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            pos_next, vel_next, acc_next = sim.step(pos_b, vel_b, acc_b)
            pos_b = pos_next
            vel_b = vel_next
            acc_b = acc_next
        t1 = time.perf_counter()
        results["numerical_verlet"][B] = ((t1 - t0) / n_repeats) * 1000.0  # ms

        # 2. Neural surrogate timing
        state_tensor = torch.randn(B, 4 * n_bodies, device=device)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_repeats):
                state_tensor = model.step(state_tensor)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        results["neural_surrogate"][B] = ((t1 - t0) / n_repeats) * 1000.0  # ms

    return results


def benchmark_rollout_horizon(
    model: NBodySurrogateMLP,
    pos0: np.ndarray,
    vel0: np.ndarray,
    masses: np.ndarray,
    steps: int = 1000,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compare sequential rollout runtime for a long horizon (e.g. 1000 steps)."""
    sim = VelocityVerletSimulator(masses=masses)
    if device is None:
        device = next(model.parameters()).device

    # Numerical
    t0 = time.perf_counter()
    _ = sim.simulate(pos0, vel0, steps=steps, track_energy=False)
    t1 = time.perf_counter()
    verlet_time = t1 - t0

    # Neural
    t0 = time.perf_counter()
    _ = model.rollout(pos0, vel0, steps=steps, device=device)
    t1 = time.perf_counter()
    neural_time = t1 - t0

    return {
        "steps": steps,
        "verlet_seconds": verlet_time,
        "neural_seconds": neural_time,
        "speedup_ratio": verlet_time / (neural_time + 1e-12),
        "verlet_us_per_step": (verlet_time / steps) * 1e6,
        "neural_us_per_step": (neural_time / steps) * 1e6,
    }
