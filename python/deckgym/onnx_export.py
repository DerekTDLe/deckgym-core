#!/usr/bin/env python3
"""
ONNX Export for MaskablePPO models with CardAttentionExtractor.

Exports the policy network to ONNX format for use in Rust self-play training
and production deployment.

Usage:
    from deckgym.onnx_export import export_policy_to_onnx
    export_policy_to_onnx(model, "frozen_opponent.onnx")
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# Constants should be fetched from Rust, but we need defaults for when module isn't built
try:
    try:
        from deckgym import Game as GameClass
    except ImportError:
        from .deckgym import PyGame as GameClass
    OBSERVATION_SIZE = GameClass.observation_size()
except (ImportError, AttributeError, ValueError):
    OBSERVATION_SIZE = 2219  # Fallback: 41 global + 18 cards * 121 features
ACTION_SPACE_SIZE = 175  # Must match src/rl/action_mask.rs


class PolicyWrapper(nn.Module):
    """
    Wrapper that extracts policy logits from SB3 MaskablePPO model.

    Takes observation and outputs action logits (before masking/softmax).
    """

    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: obs -> features -> policy latent -> action logits.

        Args:
            observation: [batch, 2849] game state

        Returns:
            action_logits: [batch, action_space_size] pre-softmax logits
        """
        # Extract features (CardAttentionExtractor)
        features = self.features_extractor(observation)

        # Get policy latent from MLP extractor
        policy_latent, _ = self.mlp_extractor(features)

        # Get action logits
        action_logits = self.action_net(policy_latent)

        return action_logits


def export_policy_to_onnx(
    model,
    output_path: str,
    observation_size: int = OBSERVATION_SIZE,
    opset_version: int = 14,
    validate: bool = True,
) -> str:
    """
    Export MaskablePPO policy to ONNX format.

    Args:
        model: MaskablePPO model instance
        output_path: Path to save ONNX file
        observation_size: Size of observation vector (default 2849)
        opset_version: ONNX opset version (14+ recommended for transformers)
        validate: Whether to validate export with onnxruntime

    Returns:
        Path to exported ONNX file
    """
    # Create wrapper
    policy = model.policy
    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Save original device and move to CPU for export
    original_device = next(policy.parameters()).device
    wrapper = wrapper.cpu()

    # Create dummy input
    dummy_input = torch.randn(1, observation_size, dtype=torch.float32)

    # Get action space size from model
    action_space_size = model.action_space.n

    # Export to ONNX
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Use legacy export mode (dynamo=False) to avoid TransformerEncoder issues
    # The new dynamo-based export doesn't support nested tensors in attention masks
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["action_logits"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action_logits": {0: "batch_size"},
            },
            dynamo=False,  # Use legacy export, required for TransformerEncoder
        )

    print(f"[ONNX] Exported policy to {output_path}")

    # Restore model to original device
    wrapper = wrapper.to(original_device)

    # Validate if requested
    if validate:
        _validate_onnx_export(output_path, wrapper, observation_size, action_space_size)

    return output_path


def _validate_onnx_export(
    onnx_path: str,
    pytorch_model: nn.Module,
    observation_size: int,
    action_space_size: int,
) -> None:
    """Validate ONNX export matches PyTorch output."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[ONNX] Skipping validation (onnx/onnxruntime not installed)")
        return

    # Check ONNX model validity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Compare outputs
    test_input = np.random.randn(4, observation_size).astype(np.float32)

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()

    # ONNX output
    try:
        # Try to use TensorRT or CUDA if available
        available_providers = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available_providers:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[ONNX] Using providers: {session.get_providers()}")
    except Exception as e:
        print(f"[ONNX] Warning: Failed to initialize session with GPU providers: {e}")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    onnx_output = session.run(None, {"observation": test_input})[0]

    # Compare
    max_diff = np.abs(pytorch_output - onnx_output).max()
    if max_diff < 1e-5:
        print(f"[ONNX] Validation passed (max diff: {max_diff:.2e})")
    else:
        print(f"[ONNX] WARNING: Output mismatch (max diff: {max_diff:.2e})")


def cleanup_old_onnx_files(directory: str, keep_latest: int = 1) -> None:
    """
    Remove old ONNX files, keeping only the most recent ones.

    Args:
        directory: Directory containing ONNX files
        keep_latest: Number of files to keep
    """
    onnx_files = sorted(
        Path(directory).glob("*.onnx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for old_file in onnx_files[keep_latest:]:
        old_file.unlink()
        print(f"[ONNX] Removed old file: {old_file}")


if __name__ == "__main__":
    # Test export with a dummy model
    import argparse

    parser = argparse.ArgumentParser(description="Export MaskablePPO to ONNX")
    parser.add_argument("--model", required=True, help="Path to SB3 checkpoint")
    parser.add_argument("--output", default="model.onnx", help="Output ONNX path")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO

    print(f"Loading model from {args.model}...")
    model = MaskablePPO.load(args.model, device="cpu")

    export_policy_to_onnx(model, args.output)
    print("Done!")
