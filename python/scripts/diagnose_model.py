#!/usr/bin/env python3
"""Clean diagnostic script for attention model."""

import argparse
import numpy as np
import torch
from sb3_contrib import MaskablePPO


def analyze_gradients(model):
    """Analyze gradient magnitudes."""
    print("\n" + "=" * 70)
    print("GRADIENT ANALYSIS")
    print("=" * 70)

    model.policy.train()

    # Create dummy input
    obs = torch.randn(1, model.observation_space.shape[0]).to(model.device)

    # Forward pass
    features = model.policy.features_extractor(obs)
    latent_pi = model.policy.mlp_extractor.forward_actor(features)
    action_logits = model.policy.action_net(latent_pi)

    # Backward pass
    loss = action_logits.sum()
    loss.backward()

    # Collect gradients
    print("\n  Gradient Norms:")
    grad_stats = {}
    for name, param in model.policy.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats[name] = grad_norm

            status = "✅"
            if grad_norm < 1e-6:
                status = "❌ VANISHING"
            elif grad_norm > 100:
                status = "⚠️  LARGE"

            print(f"    {name:55s}: {grad_norm:10.6f} {status}")

    # Check imbalance
    if grad_stats:
        max_grad = max(grad_stats.values())
        min_grad = min([v for v in grad_stats.values() if v > 1e-9])
        ratio = max_grad / min_grad if min_grad > 0 else float("inf")

        print(f"\n  Gradient Imbalance Ratio: {ratio:.1f}")
        if ratio > 1000:
            print("  ❌ CRITICAL: Ratio >1000")
        elif ratio > 100:
            print("  ⚠️  WARNING: Ratio >100")
        else:
            print("  ✅ Balanced")

        return ratio
    return None


def analyze_weights(model):
    """Analyze weight distributions."""
    print("\n" + "=" * 70)
    print("WEIGHT ANALYSIS")
    print("=" * 70)

    for name, param in model.policy.named_parameters():
        if "weight" in name and param.numel() > 100:
            weights = param.data.cpu().numpy().flatten()
            mean = weights.mean()
            std = weights.std()

            status = "✅"
            if abs(mean) > 0.1:
                status = "⚠️  High mean"
            if std < 0.01 or std > 2.0:
                status = "⚠️  Unusual std"

            print(f"  {name:55s}: μ={mean:7.4f}, σ={std:6.4f} {status}")


def analyze_attention_layers(model):
    """Analyze attention-specific layers."""
    print("\n" + "=" * 70)
    print("ATTENTION LAYER ANALYSIS")
    print("=" * 70)

    # Check if model has attention
    has_attention = hasattr(model.policy.features_extractor, "attention_layers")

    if not has_attention:
        print("  ⚠️  No attention layers found (MLP model)")
        return

    print("  ✅ Attention layers present")

    # Analyze attention weights
    for name, param in model.policy.named_parameters():
        if (
            "attention" in name.lower()
            or "q_proj" in name
            or "k_proj" in name
            or "v_proj" in name
        ):
            weights = param.data.cpu().numpy().flatten()
            print(f"  {name:55s}: μ={weights.mean():7.4f}, σ={weights.std():6.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    print("=" * 70)
    print("MODEL DIAGNOSTIC")
    print("=" * 70)
    print(f"Model: {args.model}\n")

    # Load model
    print("Loading model...")
    model = MaskablePPO.load(args.model, device="cpu")
    print(f"  Device: {model.device}")
    print(f"  Observation space: {model.observation_space.shape}")
    print(f"  Action space: {model.action_space.n}")

    # Run diagnostics
    grad_ratio = analyze_gradients(model)
    analyze_weights(model)
    analyze_attention_layers(model)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    issues = []
    if grad_ratio and grad_ratio > 1000:
        issues.append("Severe gradient imbalance")

    if issues:
        print("  ⚠️  Issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ No critical issues!")


if __name__ == "__main__":
    main()
