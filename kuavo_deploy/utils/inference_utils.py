#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLASH-inspired inference utilities for kuavo_deploy.

This module provides utilities for optimizing policy inference:
- warmup_policy: Warm up compiled policies to complete JIT compilation
- create_dummy_observation: Create dummy observations for warmup
- compile_policy: Apply torch.compile with proper settings

These utilities are inspired by the VLASH framework for VLA acceleration.
"""

import time
import logging
from typing import Dict, Any, Optional

import torch
from torch import Tensor

log = logging.getLogger("inference_utils")


def create_dummy_observation(policy_config, device: torch.device) -> Dict[str, Tensor]:
    """Create a dummy observation matching the policy's expected input shape.
    
    Args:
        policy_config: Policy configuration with input_features and image_features
        device: Target device for tensors
        
    Returns:
        Dictionary of dummy observation tensors
    """
    dummy_obs = {}
    
    # Add dummy image observations with correct shape [B, C, H, W]
    if hasattr(policy_config, 'image_features') and policy_config.image_features:
        for img_key, img_feature in policy_config.image_features.items():
            if hasattr(img_feature, 'shape'):
                shape = img_feature.shape
                if len(shape) == 3:
                    channels, height, width = shape
                else:
                    # Fallback for different shape formats
                    channels, height, width = 3, 480, 640
            else:
                channels, height, width = 3, 480, 640
                
            dummy_obs[img_key] = torch.zeros(
                (1, channels, height, width),
                dtype=torch.float32,
                device=device,
            )
    
    # Add dummy depth observations if configured
    if hasattr(policy_config, 'depth_features') and policy_config.depth_features:
        for depth_key, depth_feature in policy_config.depth_features.items():
            if hasattr(depth_feature, 'shape'):
                shape = depth_feature.shape
                if len(shape) == 3:
                    channels, height, width = shape
                else:
                    channels, height, width = 1, 480, 640
            else:
                channels, height, width = 1, 480, 640
                
            dummy_obs[depth_key] = torch.zeros(
                (1, channels, height, width),
                dtype=torch.float32,
                device=device,
            )
    
    # Add dummy state observation with correct shape [B, state_dim]
    if hasattr(policy_config, 'input_features') and policy_config.input_features:
        if "observation.state" in policy_config.input_features:
            state_feature = policy_config.input_features["observation.state"]
            if hasattr(state_feature, 'shape'):
                state_dim = state_feature.shape[0] if len(state_feature.shape) > 0 else 16
            else:
                state_dim = 16  # Default fallback
            dummy_obs["observation.state"] = torch.zeros(
                (1, state_dim),
                dtype=torch.float32,
                device=device,
            )
    
    return dummy_obs


def warmup_policy(
    policy,
    device: torch.device,
    warmup_steps: int = 3,
    dummy_obs: Optional[Dict[str, Tensor]] = None
) -> float:
    """Warm up a compiled policy to complete JIT compilation.
    
    Running a few inference passes before actual control ensures that
    torch.compile has finished optimizing the model, avoiding latency
    spikes during real operation.
    
    Args:
        policy: The policy model (possibly compiled)
        device: Target device
        warmup_steps: Number of warmup iterations
        dummy_obs: Optional pre-created dummy observation
        
    Returns:
        Total warmup time in seconds
    """
    log.info(f"Warming up policy ({warmup_steps} steps)...")
    
    # Create dummy observation if not provided
    if dummy_obs is None:
        dummy_obs = create_dummy_observation(policy.config, device)
    
    warmup_start = time.perf_counter()
    
    # Run warmup iterations to complete compilation
    for i in range(warmup_steps):
        with torch.inference_mode():
            try:
                _ = policy.select_action(dummy_obs)
            except Exception as e:
                log.warning(f"Warmup step {i+1} failed: {e}")
                # Try with predict_action_chunk if select_action fails
                try:
                    _ = policy.predict_action_chunk(dummy_obs)
                except Exception as e2:
                    log.warning(f"Warmup with predict_action_chunk also failed: {e2}")
                    break
    
    warmup_time = time.perf_counter() - warmup_start
    log.info(f"Warmup complete ({warmup_steps} steps in {warmup_time:.2f}s)")
    
    return warmup_time


def compile_policy(
    policy,
    compile_mode: str = "max-autotune",
    device: torch.device = None,
    warmup_steps: int = 3
):
    """Apply torch.compile optimization to a policy.
    
    Args:
        policy: The policy model to compile
        compile_mode: Compile mode (default/reduce-overhead/max-autotune)
        device: Target device for warmup
        warmup_steps: Number of warmup steps after compilation
        
    Returns:
        Compiled policy (may be the same object if compile fails)
    """
    try:
        # Set high precision for matmul
        torch.set_float32_matmul_precision("high")
        
        # Compile the policy
        compiled_policy = torch.compile(policy, mode=compile_mode)
        log.info(f"torch.compile applied with mode: {compile_mode}")
        
        # Warmup if device is provided
        if device is not None and warmup_steps > 0:
            warmup_policy(compiled_policy, device, warmup_steps)
        
        return compiled_policy
        
    except Exception as e:
        log.warning(f"torch.compile failed: {e}. Using original policy.")
        return policy


def estimate_inference_time(
    policy,
    device: torch.device,
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, float]:
    """Estimate average inference time for a policy.
    
    Args:
        policy: The policy model
        device: Target device
        num_runs: Number of timed runs
        warmup_runs: Number of warmup runs before timing
        
    Returns:
        Dictionary with timing statistics
    """
    dummy_obs = create_dummy_observation(policy.config, device)
    
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.inference_mode():
            _ = policy.select_action(dummy_obs)
    
    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.inference_mode():
            _ = policy.select_action(dummy_obs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "total_time": sum(times),
        "num_runs": num_runs
    }
