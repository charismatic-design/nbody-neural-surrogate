"""Numerical integrators for the 2D N-body gravitational problem.

Includes:
- VelocityVerletSimulator: Symplectic 2nd-order integrator with excellent
  long-term phase-space conservation and bounded energy oscillation.
- EulerSimulator: 1st-order explicit Euler integrator for comparison.
- SciPySimulator: High-order adaptive Runge-Kutta (RK45 / DOP853) via scipy.integrate.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from nbody.physics import (
    compute_gravitational_accelerations,
    compute_total_energy,
)


@dataclass
class SimulationResult:
    """Stores full trajectory time series and conservation diagnostics."""

    times: np.ndarray             # (T,)
    positions: np.ndarray         # (T, N, 2) or (T, B, N, 2)
    velocities: np.ndarray        # (T, N, 2) or (T, B, N, 2)
    accelerations: np.ndarray     # (T, N, 2) or (T, B, N, 2)
    total_energies: np.ndarray    # (T,) or (T, B)
    kinetic_energies: np.ndarray  # (T,) or (T, B)
    potential_energies: np.ndarray# (T,) or (T, B)
    relative_energy_errors: np.ndarray  # (T,) or (T, B)

    @property
    def max_energy_drift(self) -> float:
        """Maximum absolute relative energy error over time."""
        return float(np.max(self.relative_energy_errors))

    @property
    def final_energy_drift(self) -> float:
        """Final relative energy error at the end of the simulation."""
        return float(self.relative_energy_errors[-1])


class VelocityVerletSimulator:
    """Velocity Verlet symplectic integrator for N-body gravitational systems.

    The Velocity Verlet algorithm is time-reversible and symplectic:
        1. x(t + dt) = x(t) + v(t)*dt + 0.5 * a(t) * dt^2
        2. a(t + dt) = compute_accelerations(x(t + dt))
        3. v(t + dt) = v(t) + 0.5 * (a(t) + a(t + dt)) * dt
    """

    def __init__(
        self,
        masses: np.ndarray,
        G: float = 1.0,
        softening: float = 0.05,
        dt: float = 0.01,
    ):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.G = float(G)
        self.softening = float(softening)
        self.dt = float(dt)

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform a single Velocity Verlet integration step."""
        if accelerations is None:
            accelerations = compute_gravitational_accelerations(
                positions, self.masses, G=self.G, softening=self.softening
            )

        # 1. Update position
        pos_next = positions + velocities * self.dt + 0.5 * accelerations * (self.dt ** 2)

        # 2. Evaluate acceleration at new position
        acc_next = compute_gravitational_accelerations(
            pos_next, self.masses, G=self.G, softening=self.softening
        )

        # 3. Update velocity using average acceleration
        vel_next = velocities + 0.5 * (accelerations + acc_next) * self.dt

        return pos_next, vel_next, acc_next

    def simulate(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        steps: int,
        track_energy: bool = True,
    ) -> SimulationResult:
        """Simulate trajectory for a given number of steps.

        Parameters
        ----------
        pos0 : np.ndarray, shape (N, 2)
            Initial particle positions.
        vel0 : np.ndarray, shape (N, 2)
            Initial particle velocities.
        steps : int
            Number of integration steps.
        track_energy : bool, optional
            Whether to compute energy at every step, default True.

        Returns
        -------
        SimulationResult
        """
        n_particles = pos0.shape[0]
        pos_history = np.zeros((steps + 1, n_particles, 2), dtype=np.float64)
        vel_history = np.zeros((steps + 1, n_particles, 2), dtype=np.float64)
        acc_history = np.zeros((steps + 1, n_particles, 2), dtype=np.float64)
        times = np.arange(steps + 1, dtype=np.float64) * self.dt

        tot_e = np.zeros(steps + 1, dtype=np.float64)
        kin_e = np.zeros(steps + 1, dtype=np.float64)
        pot_e = np.zeros(steps + 1, dtype=np.float64)
        rel_err = np.zeros(steps + 1, dtype=np.float64)

        pos_cur = np.array(pos0, dtype=np.float64)
        vel_cur = np.array(vel0, dtype=np.float64)
        acc_cur = compute_gravitational_accelerations(
            pos_cur, self.masses, G=self.G, softening=self.softening
        )

        pos_history[0] = pos_cur
        vel_history[0] = vel_cur
        acc_history[0] = acc_cur

        if track_energy:
            e0, k0, p0 = compute_total_energy(
                pos_cur, vel_cur, self.masses, G=self.G, softening=self.softening
            )
            tot_e[0], kin_e[0], pot_e[0] = e0, k0, p0
            rel_err[0] = 0.0

        for t in range(steps):
            pos_cur, vel_cur, acc_cur = self.step(pos_cur, vel_cur, acc_cur)
            pos_history[t + 1] = pos_cur
            vel_history[t + 1] = vel_cur
            acc_history[t + 1] = acc_cur

            if track_energy:
                e_t, k_t, p_t = compute_total_energy(
                    pos_cur, vel_cur, self.masses, G=self.G, softening=self.softening
                )
                tot_e[t + 1], kin_e[t + 1], pot_e[t + 1] = e_t, k_t, p_t
                rel_err[t + 1] = abs((e_t - tot_e[0]) / (abs(tot_e[0]) + 1e-12))

        return SimulationResult(
            times=times,
            positions=pos_history,
            velocities=vel_history,
            accelerations=acc_history,
            total_energies=tot_e,
            kinetic_energies=kin_e,
            potential_energies=pot_e,
            relative_energy_errors=rel_err,
        )

    def simulate_batch(
        self,
        pos0_batch: np.ndarray,
        vel0_batch: np.ndarray,
        steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a batch of trajectories vectorized across batch size B.

        Parameters
        ----------
        pos0_batch : np.ndarray, shape (B, N, 2)
        vel0_batch : np.ndarray, shape (B, N, 2)
        steps : int

        Returns
        -------
        (positions, velocities) : Tuple of shape (steps + 1, B, N, 2)
        """
        B, N, _ = pos0_batch.shape
        pos_all = np.zeros((steps + 1, B, N, 2), dtype=np.float64)
        vel_all = np.zeros((steps + 1, B, N, 2), dtype=np.float64)

        pos_cur = np.array(pos0_batch, dtype=np.float64)
        vel_cur = np.array(vel0_batch, dtype=np.float64)
        acc_cur = compute_gravitational_accelerations(
            pos_cur, self.masses, G=self.G, softening=self.softening
        )

        pos_all[0] = pos_cur
        vel_all[0] = vel_cur

        for t in range(steps):
            pos_cur, vel_cur, acc_cur = self.step(pos_cur, vel_cur, acc_cur)
            pos_all[t + 1] = pos_cur
            vel_all[t + 1] = vel_cur

        return pos_all, vel_all


class EulerSimulator:
    """Standard forward Euler 1st-order integrator for comparison."""

    def __init__(
        self,
        masses: np.ndarray,
        G: float = 1.0,
        softening: float = 0.05,
        dt: float = 0.01,
    ):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.G = float(G)
        self.softening = float(softening)
        self.dt = float(dt)

    def step(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        acc = compute_gravitational_accelerations(
            positions, self.masses, G=self.G, softening=self.softening
        )
        pos_next = positions + velocities * self.dt
        vel_next = velocities + acc * self.dt
        return pos_next, vel_next

    def simulate(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        steps: int,
    ) -> SimulationResult:
        n_particles = pos0.shape[0]
        pos_history = np.zeros((steps + 1, n_particles, 2), dtype=np.float64)
        vel_history = np.zeros((steps + 1, n_particles, 2), dtype=np.float64)
        acc_history = np.zeros((steps + 1, n_particles, 2), dtype=np.float64)
        times = np.arange(steps + 1, dtype=np.float64) * self.dt

        tot_e = np.zeros(steps + 1, dtype=np.float64)
        kin_e = np.zeros(steps + 1, dtype=np.float64)
        pot_e = np.zeros(steps + 1, dtype=np.float64)
        rel_err = np.zeros(steps + 1, dtype=np.float64)

        pos_cur = np.array(pos0, dtype=np.float64)
        vel_cur = np.array(vel0, dtype=np.float64)

        pos_history[0] = pos_cur
        vel_history[0] = vel_cur
        acc_history[0] = compute_gravitational_accelerations(
            pos_cur, self.masses, G=self.G, softening=self.softening
        )

        e0, k0, p0 = compute_total_energy(
            pos_cur, vel_cur, self.masses, G=self.G, softening=self.softening
        )
        tot_e[0], kin_e[0], pot_e[0] = e0, k0, p0
        rel_err[0] = 0.0

        for t in range(steps):
            pos_cur, vel_cur = self.step(pos_cur, vel_cur)
            pos_history[t + 1] = pos_cur
            vel_history[t + 1] = vel_cur
            acc_cur = compute_gravitational_accelerations(
                pos_cur, self.masses, G=self.G, softening=self.softening
            )
            acc_history[t + 1] = acc_cur

            e_t, k_t, p_t = compute_total_energy(
                pos_cur, vel_cur, self.masses, G=self.G, softening=self.softening
            )
            tot_e[t + 1], kin_e[t + 1], pot_e[t + 1] = e_t, k_t, p_t
            rel_err[t + 1] = abs((e_t - tot_e[0]) / (abs(tot_e[0]) + 1e-12))

        return SimulationResult(
            times=times,
            positions=pos_history,
            velocities=vel_history,
            accelerations=acc_history,
            total_energies=tot_e,
            kinetic_energies=kin_e,
            potential_energies=pot_e,
            relative_energy_errors=rel_err,
        )


class SciPySimulator:
    """Continuous ODE simulator using scipy.integrate.solve_ivp (RK45/DOP853)."""

    def __init__(
        self,
        masses: np.ndarray,
        G: float = 1.0,
        softening: float = 0.05,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-8,
    ):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.G = float(G)
        self.softening = float(softening)
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def _ode_func(self, t: float, y: np.ndarray) -> np.ndarray:
        # y is 4*N array: [pos_x_all, pos_y_all, vel_x_all, vel_y_all] or interleaved
        n = len(self.masses)
        pos = y[: 2 * n].reshape(n, 2)
        vel = y[2 * n :].reshape(n, 2)
        acc = compute_gravitational_accelerations(
            pos, self.masses, G=self.G, softening=self.softening
        )
        return np.concatenate([vel.flatten(), acc.flatten()])

    def simulate(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        t_span: Tuple[float, float],
        eval_times: np.ndarray,
    ) -> SimulationResult:
        n = len(self.masses)
        y0 = np.concatenate([pos0.flatten(), vel0.flatten()])
        sol = solve_ivp(
            self._ode_func,
            t_span,
            y0,
            t_eval=eval_times,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        t_steps = len(sol.t)
        pos_hist = sol.y[: 2 * n, :].T.reshape(t_steps, n, 2)
        vel_hist = sol.y[2 * n :, :].T.reshape(t_steps, n, 2)
        acc_hist = np.zeros_like(pos_hist)

        tot_e = np.zeros(t_steps)
        kin_e = np.zeros(t_steps)
        pot_e = np.zeros(t_steps)
        rel_err = np.zeros(t_steps)

        for i in range(t_steps):
            acc_hist[i] = compute_gravitational_accelerations(
                pos_hist[i], self.masses, G=self.G, softening=self.softening
            )
            e, k, p = compute_total_energy(
                pos_hist[i], vel_hist[i], self.masses, G=self.G, softening=self.softening
            )
            tot_e[i], kin_e[i], pot_e[i] = e, k, p
            rel_err[i] = abs((e - tot_e[0]) / (abs(tot_e[0]) + 1e-12))

        return SimulationResult(
            times=sol.t,
            positions=pos_hist,
            velocities=vel_hist,
            accelerations=acc_hist,
            total_energies=tot_e,
            kinetic_energies=kin_e,
            potential_energies=pot_e,
            relative_energy_errors=rel_err,
        )
