"""PyTorch Neural Surrogate Model for N-body dynamical systems."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual Feed-Forward Block with LayerNorm and GELU activation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(x + self.net(x)))


class NBodySurrogateMLP(nn.Module):
    """Neural Surrogate predicting state residuals delta_s = [delta_x, delta_v].

    Architecture:
      Input (4*N) -> Linear Projection -> K ResBlocks -> Output Projection (4*N)

    Equipped with an autoregressive rollout loop for short-to-medium horizon evaluation.
    """

    def __init__(
        self,
        n_bodies: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_bodies = n_bodies
        self.state_dim = 4 * n_bodies  # [x1, y1, ..., xN, yN, vx1, vy1, ..., vxN, vyN]

        self.input_proj = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout=dropout) for _ in range(n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.state_dim),
        )

        # Optional normalization buffers
        self.register_buffer("input_mean", torch.zeros(self.state_dim))
        self.register_buffer("input_std", torch.ones(self.state_dim))
        self.register_buffer("target_mean", torch.zeros(self.state_dim))
        self.register_buffer("target_std", torch.ones(self.state_dim))
        self.is_normalized = False

    def set_normalization(self, stats: Dict[str, np.ndarray]):
        """Store dataset mean/std for automatic input/target standardization."""
        self.input_mean.copy_(torch.from_numpy(stats["input_mean"].astype(np.float32)))
        self.input_std.copy_(torch.from_numpy(stats["input_std"].astype(np.float32)))
        self.target_mean.copy_(torch.from_numpy(stats["target_mean"].astype(np.float32)))
        self.target_std.copy_(torch.from_numpy(stats["target_std"].astype(np.float32)))
        self.is_normalized = True

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict the state change delta_s given state s.

        Parameters
        ----------
        state : torch.Tensor, shape (..., 4 * N)

        Returns
        -------
        delta_s : torch.Tensor, shape (..., 4 * N)
        """
        orig_state = state
        if self.is_normalized:
            state = (state - self.input_mean) / self.input_std

        h = self.input_proj(state)
        for block in self.blocks:
            h = block(h)
        delta = self.output_proj(h)

        if self.is_normalized:
            delta = delta * self.target_std + self.target_mean

        return delta

    def step(self, state: torch.Tensor) -> torch.Tensor:
        """Predict next state s_{t+1} = s_t + delta_s."""
        delta = self.forward(state)
        return state + delta

    @torch.no_grad()
    def rollout(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        steps: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Autoregressively roll out the surrogate model for `steps` iterations.

        Parameters
        ----------
        pos0 : np.ndarray, shape (N, 2)
        vel0 : np.ndarray, shape (N, 2)
        steps : int
        device : torch.device, optional

        Returns
        -------
        (positions, velocities) : Tuple[np.ndarray, np.ndarray]
            positions: (steps + 1, N, 2)
            velocities: (steps + 1, N, 2)
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        n = pos0.shape[0]
        # Initial state flat vector: (4 * N,)
        state_flat = np.concatenate([pos0.flatten(), vel0.flatten()]).astype(np.float32)
        s_tensor = torch.from_numpy(state_flat).unsqueeze(0).to(device)  # (1, 4N)

        pos_hist = [pos0.copy()]
        vel_hist = [vel0.copy()]

        for _ in range(steps):
            s_tensor = self.step(s_tensor)
            s_np = s_tensor.squeeze(0).cpu().numpy()
            pos = s_np[: 2 * n].reshape(n, 2)
            vel = s_np[2 * n :].reshape(n, 2)
            pos_hist.append(pos)
            vel_hist.append(vel)

        return np.array(pos_hist, dtype=np.float64), np.array(vel_hist, dtype=np.float64)
