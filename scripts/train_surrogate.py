"""CLI to train the PyTorch N-body neural surrogate model."""

import argparse
from nbody.train import train_surrogate


def main():
    parser = argparse.ArgumentParser(description="Train N-body Neural Surrogate")
    parser.add_argument("--bodies", type=int, default=3, help="Number of bodies")
    parser.add_argument("--train-trajs", type=int, default=100, help="Train trajectories")
    parser.add_argument("--val-trajs", type=int, default=25, help="Val trajectories")
    parser.add_argument("--steps", type=int, default=250, help="Steps per trajectory")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="ResMLP layers")
    parser.add_argument("--save-path", type=str, default="checkpoints/best_surrogate.pt")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model, history = train_surrogate(
        n_bodies=args.bodies,
        n_train_trajectories=args.train_trajs,
        n_val_trajectories=args.val_trajs,
        steps_per_trajectory=args.steps,
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        save_path=args.save_path,
        device=args.device,
        seed=args.seed,
    )
    print(f"\n[Success] Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
