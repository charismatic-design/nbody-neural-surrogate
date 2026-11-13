"""Unit tests for neural surrogate model and rollout dynamics."""

import numpy as np
import pytest
import torch

from nbody.model import NBodySurrogateMLP


def test_model_forward_and_residual_shapes():
    n_bodies = 3
    model = NBodySurrogateMLP(n_bodies=n_bodies, hidden_dim=64, n_layers=2)

    batch_size = 8
    dummy_input = torch.randn(batch_size, 4 * n_bodies)

    delta = model(dummy_input)
    assert delta.shape == (batch_size, 4 * n_bodies)

    next_state = model.step(dummy_input)
    assert next_state.shape == (batch_size, 4 * n_bodies)
    np.testing.assert_allclose(next_state.detach().numpy(), (dummy_input + delta).detach().numpy())


def test_autoregressive_rollout_shapes():
    n_bodies = 4
    model = NBodySurrogateMLP(n_bodies=n_bodies, hidden_dim=64, n_layers=2)

    pos0 = np.random.randn(n_bodies, 2)
    vel0 = np.random.randn(n_bodies, 2)
    steps = 50

    pos_hist, vel_hist = model.rollout(pos0, vel0, steps=steps)

    assert pos_hist.shape == (steps + 1, n_bodies, 2)
    assert vel_hist.shape == (steps + 1, n_bodies, 2)
    np.testing.assert_allclose(pos_hist[0], pos0)
    np.testing.assert_allclose(vel_hist[0], vel0)
