"""Quick CLI to run a 2D N-body gravitational simulation and verify energy conservation."""

import argparse
import numpy as np

from nbody.physics import compute_center_of_mass, compute_total_momentum
from nbody.simulator import EulerSimulator, VelocityVerletSimulator
from nbody.dataset import sample_initial_conditions


def main():
    parser = argparse.ArgumentParser(description="Simulate 2D N-body system")
    parser.add_argument("--bodies", type=int, default=3, help="Number of particles")
    parser.add_argument("--steps", type=int, default=500, help="Integration steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step dt")
    parser.add_argument("--scenario", type=str, default="figure_eight", choices=["figure_eight", "random_bounded", "central_mass"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pos0, vel0, masses = sample_initial_conditions(n_bodies=args.bodies, scenario=args.scenario, seed=args.seed)
    print(f"=== N-body Simulation ({args.scenario}, N={args.bodies}, dt={args.dt}, steps={args.steps}) ===")
    print(f"Initial Center of Mass: {compute_center_of_mass(pos0, masses)}")
    print(f"Initial Linear Momentum: {compute_total_momentum(vel0, masses)}")

    vv_sim = VelocityVerletSimulator(masses=masses, dt=args.dt)
    vv_res = vv_sim.simulate(pos0, vel0, steps=args.steps)

    eu_sim = EulerSimulator(masses=masses, dt=args.dt)
    eu_res = eu_sim.simulate(pos0, vel0, steps=args.steps)

    print("\n--- Conservation Results ---")
    print(f"Initial Total Energy E0: {vv_res.total_energies[0]:.6f}")
    print(f"Velocity Verlet Final E:  {vv_res.total_energies[-1]:.6f} | Max Rel Error: {vv_res.max_energy_drift:.4e}")
    print(f"Euler Final E:            {eu_res.total_energies[-1]:.6f} | Max Rel Error: {eu_res.max_energy_drift:.4e}")
    ratio = eu_res.max_energy_drift / (vv_res.max_energy_drift + 1e-15)
    print(f"Verlet is {ratio:.1f}x more energy-stable than explicit Euler!")


if __name__ == "__main__":
    main()
