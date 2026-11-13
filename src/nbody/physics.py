"""Vectorized 2D N-body gravitational physics engine.

Implements pairwise gravitational acceleration computation using Plummer
softening, kinetic and potential energy tracking, and linear momentum conservation.
"""

from typing import Tuple, Union
import numpy as np


def compute_gravitational_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    softening: float = 0.05,
) -> np.ndarray:
    """Compute gravitational accelerations on all particles using vectorized linear algebra.

    Uses Newton's law of universal gravitation with Plummer softening:
        a_i = G * sum_{j != i} m_j * (x_j - x_i) / (||x_j - x_i||^2 + eps^2)^(3/2)

    Supports both single state (N, 2) and batched states (B, N, 2) or (..., N, 2).

    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (..., N, 2).
    masses : np.ndarray
        Particle masses, shape (..., N) or (N,).
    G : float, optional
        Gravitational constant, default 1.0.
    softening : float, optional
        Plummer softening length epsilon to prevent singularities, default 0.05.

    Returns
    -------
    accelerations : np.ndarray
        Gravitational accelerations, shape (..., N, 2).
    """
    pos = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    n_particles = pos.shape[-2]

    # Pairwise displacement: diff[..., i, j, :] = pos[j] - pos[i]
    # shape: (..., N, N, 2)
    diff = pos[..., None, :, :] - pos[..., :, None, :]

    # Pairwise squared distances: shape (..., N, N)
    dist_sq = np.sum(diff**2, axis=-1)

    # Softened distance cubed: (r_ij^2 + eps^2)^(3/2)
    softened_dist_cubed = (dist_sq + softening**2) ** 1.5

    # Broadcast masses to (..., 1, N, 1) so m[j] scales diff[..., i, j, :]
    if m.ndim == 1:
        m_broadcast = m[None, :, None]
    else:
        m_broadcast = m[..., None, :, None]

    # weights: shape (..., N, N, 1) = m_j / softened_dist_cubed[..., i, j, None]
    weights = m_broadcast / softened_dist_cubed[..., None]

    # Exclude self-interaction (diagonal i == j) using broadcastable boolean mask
    mask_not_diag = (~np.eye(n_particles, dtype=bool))[..., None]  # shape (N, N, 1)
    weights = weights * mask_not_diag

    # Accelerations on body i: sum over all j (axis=-2)
    accelerations = G * np.sum(weights * diff, axis=-2)
    return accelerations


def compute_kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> Union[float, np.ndarray]:
    """Compute total kinetic energy of the N-body system.

    T = 0.5 * sum_i m_i * ||v_i||^2

    Parameters
    ----------
    velocities : np.ndarray
        Particle velocities, shape (..., N, 2).
    masses : np.ndarray
        Particle masses, shape (..., N) or (N,).

    Returns
    -------
    kinetic_energy : float or np.ndarray
        Total kinetic energy, shape (...) or scalar float.
    """
    vel = np.asarray(velocities, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    v_sq = np.sum(vel**2, axis=-1)  # shape (..., N)

    if m.ndim < v_sq.ndim:
        ke = 0.5 * np.sum(m * v_sq, axis=-1)
    else:
        ke = 0.5 * np.sum(m * v_sq, axis=-1)

    if np.ndim(ke) == 0:
        return float(ke)
    return ke


def compute_potential_energy(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    softening: float = 0.05,
) -> Union[float, np.ndarray]:
    """Compute gravitational potential energy with Plummer softening.

    U = -0.5 * G * sum_{i != j} (m_i * m_j) / sqrt(||x_i - x_j||^2 + eps^2)

    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (..., N, 2).
    masses : np.ndarray
        Particle masses, shape (..., N) or (N,).
    G : float, optional
        Gravitational constant, default 1.0.
    softening : float, optional
        Plummer softening length, default 0.05.

    Returns
    -------
    potential_energy : float or np.ndarray
        Total potential energy, shape (...) or scalar float.
    """
    pos = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    n_particles = pos.shape[-2]

    # Pairwise difference and distance
    diff = pos[..., None, :, :] - pos[..., :, None, :]  # shape (..., N, N, 2)
    dist_sq = np.sum(diff**2, axis=-1)  # shape (..., N, N)
    softened_dist = np.sqrt(dist_sq + softening**2)

    # Compute outer product of masses m_i * m_j: shape (..., N, N)
    if m.ndim == 1:
        m_matrix = np.outer(m, m)
    else:
        m_matrix = m[..., :, None] * m[..., None, :]

    # Inverse softened distances with zeroed diagonal to exclude self-energy
    inv_dist = 1.0 / softened_dist
    mask_not_diag = ~np.eye(n_particles, dtype=bool)  # shape (N, N)
    inv_dist = inv_dist * mask_not_diag

    pe = -0.5 * G * np.sum(m_matrix * inv_dist, axis=(-1, -2))
    if np.ndim(pe) == 0:
        return float(pe)
    return pe


def compute_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    softening: float = 0.05,
) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute kinetic, potential, and total energy: E = T + U.

    Returns
    -------
    (total_energy, kinetic_energy, potential_energy)
    """
    ke = compute_kinetic_energy(velocities, masses)
    pe = compute_potential_energy(positions, masses, G=G, softening=softening)
    total = ke + pe
    return total, ke, pe


def compute_center_of_mass(
    positions: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Compute center of mass position vector R_cm = sum(m_i * x_i) / sum(m_i)."""
    pos = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    if m.ndim == 1:
        total_m = np.sum(m)
        return np.sum(pos * m[:, None], axis=-2) / total_m
    else:
        total_m = np.sum(m, axis=-1, keepdims=True)
        return np.sum(pos * m[..., :, None], axis=-2) / total_m


def compute_total_momentum(
    velocities: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """Compute total linear momentum P = sum(m_i * v_i)."""
    vel = np.asarray(velocities, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    if m.ndim == 1:
        return np.sum(vel * m[:, None], axis=-2)
    else:
        return np.sum(vel * m[..., :, None], axis=-2)
