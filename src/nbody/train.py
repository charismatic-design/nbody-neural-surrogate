"""Training pipeline for N-body neural surrogate model."""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nbody.dataset import NBodyDataset, generate_trajectory_dataset
from nbody.model import NBodySurrogateMLP


def train_surrogate(
    n_bodies: int = 3,
    n_train_trajectories: int = 80,
    n_val_trajectories: int = 20,
    steps_per_trajectory: int = 250,
    dt: float = 0.01,
    G: float = 1.0,
    softening: float = 0.05,
    hidden_dim: int = 256,
    n_layers: int = 4,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 40,
    normalize: bool = True,
    device: Optional[str] = None,
    save_path: Optional[str] = "checkpoints/best_surrogate.pt",
    seed: int = 42,
) -> Tuple[NBodySurrogateMLP, Dict[str, list]]:
    """Train the N-body neural surrogate model on simulated trajectory transitions.

    Parameters
    ----------
    n_bodies : int
        Number of particles.
    n_train_trajectories : int
        Number of distinct initial condition simulations for training.
    n_val_trajectories : int
        Number of distinct simulations for validation.
    steps_per_trajectory : int
        Number of time steps per simulated trajectory.
    dt : float
        Integration time step.
    G : float
        Gravitational constant.
    softening : float
        Plummer softening parameter.
    hidden_dim : int
        Hidden dimension of ResMLP.
    n_layers : int
        Number of residual blocks.
    batch_size : int
        Batch size for training.
    lr : float
        Initial learning rate.
    epochs : int
        Number of training epochs.
    normalize : bool
        Whether to z-score normalize inputs and targets.
    device : str, optional
        'cuda', 'mps', or 'cpu'. Auto-detects if None.
    save_path : str, optional
        Where to save the best model weights.
    seed : int
        Random seed.

    Returns
    -------
    (model, history)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    print(f"[Training] Using device: {dev}")
    print(f"[Training] Generating {n_train_trajectories} train trajectories...")
    train_data = generate_trajectory_dataset(
        n_trajectories=n_train_trajectories,
        steps_per_trajectory=steps_per_trajectory,
        n_bodies=n_bodies,
        dt=dt,
        G=G,
        softening=softening,
        seed=seed,
    )

    print(f"[Training] Generating {n_val_trajectories} val trajectories...")
    val_data = generate_trajectory_dataset(
        n_trajectories=n_val_trajectories,
        steps_per_trajectory=steps_per_trajectory,
        n_bodies=n_bodies,
        dt=dt,
        G=G,
        softening=softening,
        seed=seed + 9999,
    )

    train_ds = NBodyDataset(
        train_data["positions"],
        train_data["velocities"],
        normalize=normalize,
    )
    val_ds = NBodyDataset(
        val_data["positions"],
        val_data["velocities"],
        normalize=normalize,
        stats=train_ds.get_stats() if normalize else None,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = NBodySurrogateMLP(
        n_bodies=n_bodies,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(dev)

    if normalize:
        model.set_normalization(train_ds.get_stats())

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    best_val_loss = float("inf")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f"[Training] Training {epochs} epochs on {len(train_ds)} transition samples...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for states, targets in train_loader:
            states = states.to(dev)
            targets = targets.to(dev)

            optimizer.zero_grad()
            preds = model(states)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for states, targets in val_loader:
                states = states.to(dev)
                targets = targets.to(dev)
                preds = model(states)
                v_loss = criterion(preds, targets)
                val_loss += v_loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(1, n_val_batches)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_path:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "stats": train_ds.get_stats() if normalize else None,
                        "epoch": epoch,
                        "val_loss": avg_val_loss,
                        "n_bodies": n_bodies,
                        "hidden_dim": hidden_dim,
                        "n_layers": n_layers,
                    },
                    save_path,
                )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {avg_train_loss:.6e} | "
                f"Val Loss: {avg_val_loss:.6e} | "
                f"Best Val: {best_val_loss:.6e} | LR: {current_lr:.2e}"
            )

    print(f"[Training] Complete. Best validation loss: {best_val_loss:.6e}")
    return model, history
