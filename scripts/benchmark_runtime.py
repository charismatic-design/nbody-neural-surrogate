"""Benchmark script comparing runtime between Numerical Verlet and Neural Surrogate."""

import argparse
import os
import numpy as np
import torch

from nbody.benchmark import benchmark_rollout_horizon, benchmark_step_latencies
from nbody.dataset import sample_initial_conditions
from nbody.model import NBodySurrogateMLP
from nbody.train import train_surrogate


def main():
    parser = argparse.ArgumentParser(description="Benchmark N-body simulation vs Neural surrogate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_surrogate.pt")
    parser.add_argument("--bodies", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Benchmarking on device: {device}")

    # Load or train model
    model = NBodySurrogateMLP(n_bodies=args.bodies, hidden_dim=256, n_layers=4).to(device)
    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("stats") is not None:
                model.set_normalization(ckpt["stats"])
            print(f"Loaded checkpoint from {args.checkpoint}")
        except Exception as e:
            print(f"Could not load checkpoint ({e}), initializing fresh model.")
    else:
        print(f"Checkpoint {args.checkpoint} not found. Training a quick surrogate model...")
        model, _ = train_surrogate(
            n_bodies=args.bodies,
            epochs=25,
            save_path=args.checkpoint,
            device=device.type,
        )

    pos0, vel0, masses = sample_initial_conditions(n_bodies=args.bodies, scenario="figure_eight")

    print("\n--- Horizon Rollout Benchmark (1,000 steps) ---")
    rollout_res = benchmark_rollout_horizon(model, pos0, vel0, masses, steps=args.steps, device=device)
    print(f"Numerical Velocity Verlet: {rollout_res['verlet_seconds']:.4f} s ({rollout_res['verlet_us_per_step']:.2f} µs/step)")
    print(f"Neural Surrogate:          {rollout_res['neural_seconds']:.4f} s ({rollout_res['neural_us_per_step']:.2f} µs/step)")
    print(f"Speedup / Ratio:           {rollout_res['speedup_ratio']:.2f}x")

    print("\n--- Batched Step Latency Benchmark (Per-step time in ms) ---")
    batch_sizes = [1, 16, 64, 256, 1024]
    latencies = benchmark_step_latencies(model, n_bodies=args.bodies, batch_sizes=batch_sizes, device=device)

    print(f"{'Batch Size':<12} | {'Verlet (ms)':<14} | {'Neural (ms)':<14} | {'Throughput (Verlet)':<20} | {'Throughput (Neural)':<20}")
    print("-" * 88)
    for B in batch_sizes:
        v_ms = latencies["numerical_verlet"][B]
        n_ms = latencies["neural_surrogate"][B]
        v_thru = (B / (v_ms / 1000.0))
        n_thru = (B / (n_ms / 1000.0))
        print(f"{B:<12} | {v_ms:<14.4f} | {n_ms:<14.4f} | {v_thru:<20.1f} | {n_thru:<20.1f}")


if __name__ == "__main__":
    main()
