#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Async Inference Manager for kuavo_deploy.

This module implements asynchronous action chunk execution inspired by VLASH framework.
The key innovation is future-state-aware prediction that overlaps inference with execution.

Key concepts:
- Action Chunking: Predict multiple actions at once, execute them sequentially
- Async Inference: Start computing next chunk before current chunk finishes
- Future-State-Awareness: Use predicted end state for next inference

Usage:
    async_manager = AsyncInferenceManager(policy, n_action_steps=20, overlap_steps=4)
    
    while not done:
        action = async_manager.get_action(observation, preprocessor, postprocessor)
        observation, reward, done, info = env.step(action)
"""

import logging
import time
from copy import copy
from typing import Optional, Dict, Callable, Any

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger("async_manager")


class AsyncInferenceManager:
    """Manages asynchronous action chunk execution for efficient inference.
    
    This class implements the VLASH async inference strategy:
    1. Execute actions from the current chunk while preparing the next
    2. Use future state awareness by conditioning on predicted end state
    3. Overlap inference with execution to hide latency
    
    The execution timeline looks like:
    
        Chunk N:     [action_0, action_1, ..., action_{n-overlap}, ..., action_{n-1}]
                                                    ^
                                                    |-- Start inference for Chunk N+1
                                                        (using predicted state at action_{n-1})
        Chunk N+1:   [action_0, action_1, ...]
                     ^
                     |-- Switch to new chunk when Chunk N completes
    
    Attributes:
        policy: The trained policy for action prediction.
        n_action_steps: Number of actions per chunk.
        overlap_steps: Steps before chunk end to start next inference.
        current_chunk: Currently executing action chunk (numpy array).
        next_chunk: Pre-computed next chunk (torch tensor, pending transfer).
        chunk_index: Current position within the executing chunk.
        device: Torch device for inference.
        use_future_state: Whether to use future state awareness.
    """
    
    def __init__(
        self,
        policy,
        n_action_steps: int,
        overlap_steps: int = 4,
        device: torch.device = None,
        use_future_state: bool = True,
    ):
        """Initialize the async manager.
        
        Args:
            policy: Trained policy for action prediction.
            n_action_steps: Number of actions to execute per chunk.
            overlap_steps: Number of steps before chunk end to start next inference.
                          Higher values give more time for inference but uses
                          slightly older observations.
            device: Torch device for inference.
            use_future_state: Whether to use future state awareness.
        """
        self.policy = policy
        self.n_action_steps = n_action_steps
        self.overlap_steps = overlap_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_future_state = use_future_state
        
        # Chunk state management
        self.current_chunk: Optional[np.ndarray] = None  # Currently executing (on CPU)
        self.next_chunk: Optional[Tensor] = None         # Pre-computed (on GPU)
        self.chunk_index = 0  # Position within current chunk
        
        # Timing statistics
        self.inference_times = []
        self.total_inferences = 0
        
        # Validate configuration
        if self.n_action_steps < self.overlap_steps:
            raise ValueError(
                f"n_action_steps ({n_action_steps}) must be >= overlap_steps ({overlap_steps})"
            )
        if self.overlap_steps < 0:
            raise ValueError(f"overlap_steps must be non-negative, got {overlap_steps}")
        
        log.info(f"AsyncInferenceManager initialized: n_action_steps={n_action_steps}, "
                 f"overlap_steps={overlap_steps}, use_future_state={use_future_state}")

    def is_running(self) -> bool:
        """Check if the manager has any chunks to execute.
        
        Returns:
            True if there's a current or pending chunk, False otherwise.
        """
        return (self.current_chunk is not None) or (self.next_chunk is not None)

    def should_switch_chunk(self) -> bool:
        """Check if it's time to switch to the next chunk.
        
        Returns:
            True if at the beginning of a new chunk cycle (index == 0).
        """
        return self.chunk_index == 0

    def should_launch_next_inference(self) -> bool:
        """Check if it's time to start computing the next chunk.
        
        The next inference is launched `overlap_steps` before the current
        chunk ends, allowing inference to happen in parallel with execution.
        
        Returns:
            True if at the trigger point for next inference.
        """
        return self.chunk_index == self.n_action_steps - self.overlap_steps

    def should_fetch_observation(self) -> bool:
        """Check if a fresh observation is needed.
        
        Observations are fetched:
        1. At startup (not running yet)
        2. When launching next inference (need current state)
        
        Returns:
            True if observation should be captured this step.
        """
        return (not self.is_running()) or self.should_launch_next_inference()

    def _launch_inference(
        self,
        observation: Dict[str, Any],
        preprocessor: Optional[Callable] = None,
        apply_future_state: bool = False,
    ) -> Tensor:
        """Compute an action chunk using the policy.
        
        Implements future state awareness: if we have a current chunk and
        apply_future_state is True, use the final action as the observation
        state (predicting where the robot will be when this chunk finishes).
        
        Args:
            observation: Current observation dictionary (numpy arrays).
            preprocessor: Optional preprocessor to apply to observation.
            apply_future_state: Whether to apply future state awareness.
            
        Returns:
            Predicted action chunk as a torch tensor [n_action_steps, action_dim].
        """
        start_time = time.perf_counter()
        
        # Make a copy to avoid modifying the original
        obs = copy(observation)
        
        # Future state awareness: use predicted end state instead of current state
        if apply_future_state and self.use_future_state and self.current_chunk is not None:
            last_action = self.current_chunk[self.n_action_steps - 1]
            if "observation.state" in obs:
                # Replace current state with predicted future state
                obs["observation.state"] = torch.tensor(
                    last_action, 
                    dtype=torch.float32, 
                    device=self.device
                ).unsqueeze(0)
                log.debug("Applied future state awareness")
        
        # Apply preprocessor if provided
        if preprocessor is not None:
            obs = preprocessor(obs)
        
        with torch.inference_mode():
            # Get full action chunk from policy
            action_chunk = self.policy.predict_action_chunk(obs)
        
        # Record timing
        inference_time = time.perf_counter() - start_time
        self.inference_times.append(inference_time)
        self.total_inferences += 1
        
        log.debug(f"Inference completed in {inference_time:.3f}s")
        
        # Return chunk (remove batch dimension if present)
        if action_chunk.dim() == 3:
            return action_chunk.squeeze(0)  # [chunk_size, action_dim]
        return action_chunk

    def get_action(
        self,
        observation: Dict[str, Any],
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
    ) -> Tensor:
        """Get the next action to execute.
        
        This is the main interface called each control loop iteration.
        It manages chunk transitions and triggers async inference.
        
        Args:
            observation: Current observation dictionary (raw, before preprocessing).
            preprocessor: Optional preprocessor for observations.
            postprocessor: Optional postprocessor for actions.
            
        Returns:
            Action tensor for the robot to execute.
        """
        # Bootstrap: compute first chunk synchronously
        if not self.is_running():
            log.info("Bootstrap: computing first action chunk...")
            action_chunk = self._launch_inference(observation, preprocessor, apply_future_state=False)
            self.current_chunk = action_chunk.cpu().numpy()
            self.chunk_index = 0
        
        # Chunk transition: move pre-computed next chunk to current
        elif self.should_switch_chunk():
            if self.next_chunk is not None:
                self.current_chunk = self.next_chunk.cpu().numpy()
                self.next_chunk = None
                log.debug("Switched to pre-computed chunk")
            else:
                # Fallback: compute chunk synchronously if next_chunk wasn't ready
                log.warning("Next chunk not ready, computing synchronously...")
                action_chunk = self._launch_inference(observation, preprocessor, apply_future_state=False)
                self.current_chunk = action_chunk.cpu().numpy()

        # Async inference: start computing next chunk in advance
        if self.should_launch_next_inference() and self.next_chunk is None:
            log.debug(f"Launching async inference at step {self.chunk_index}")
            self.next_chunk = self._launch_inference(
                observation, 
                preprocessor, 
                apply_future_state=True
            )

        # Get action at current index
        action = self.current_chunk[self.chunk_index]
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Apply postprocessor if provided
        if postprocessor is not None:
            action_tensor = postprocessor(action_tensor)
        
        # Advance index
        self.chunk_index = (self.chunk_index + 1) % self.n_action_steps
        
        # Clear current chunk when we've used all actions
        if self.chunk_index == 0:
            self.current_chunk = None
        
        return action_tensor

    def reset(self):
        """Reset the manager state. Call when starting a new episode."""
        self.current_chunk = None
        self.next_chunk = None
        self.chunk_index = 0
        self.inference_times = []
        self.total_inferences = 0
        self.policy.reset()
        log.info("AsyncInferenceManager reset")

    def get_statistics(self) -> Dict[str, float]:
        """Get inference timing statistics.
        
        Returns:
            Dictionary with timing statistics.
        """
        if not self.inference_times:
            return {
                "mean_inference_time": 0.0,
                "min_inference_time": 0.0,
                "max_inference_time": 0.0,
                "total_inferences": 0,
            }
        
        return {
            "mean_inference_time": sum(self.inference_times) / len(self.inference_times),
            "min_inference_time": min(self.inference_times),
            "max_inference_time": max(self.inference_times),
            "total_inferences": self.total_inferences,
        }


class SyncInferenceManager:
    """Synchronous inference manager for comparison/fallback.
    
    This implements the traditional approach where inference happens
    synchronously at each step (or when the action queue is empty).
    """
    
    def __init__(
        self,
        policy,
        n_action_steps: int = 1,
        device: torch.device = None,
    ):
        """Initialize synchronous inference manager.
        
        Args:
            policy: Trained policy for action prediction.
            n_action_steps: Number of actions to cache per inference.
            device: Torch device for inference.
        """
        self.policy = policy
        self.n_action_steps = n_action_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.action_queue = []
        self.queue_index = 0
        
        # Timing statistics
        self.inference_times = []
        self.total_inferences = 0

    def get_action(
        self,
        observation: Dict[str, Any],
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
    ) -> Tensor:
        """Get the next action synchronously.
        
        Args:
            observation: Current observation dictionary.
            preprocessor: Optional preprocessor for observations.
            postprocessor: Optional postprocessor for actions.
            
        Returns:
            Action tensor for the robot to execute.
        """
        # Refill queue if empty
        if self.queue_index >= len(self.action_queue):
            start_time = time.perf_counter()
            
            obs = observation
            if preprocessor is not None:
                obs = preprocessor(obs)
            
            with torch.inference_mode():
                action_chunk = self.policy.predict_action_chunk(obs)
            
            # Extract n_action_steps actions
            if action_chunk.dim() == 3:
                action_chunk = action_chunk.squeeze(0)
            
            self.action_queue = [
                action_chunk[i] for i in range(min(self.n_action_steps, action_chunk.shape[0]))
            ]
            self.queue_index = 0
            
            inference_time = time.perf_counter() - start_time
            self.inference_times.append(inference_time)
            self.total_inferences += 1
        
        # Get action from queue
        action = self.action_queue[self.queue_index]
        self.queue_index += 1
        
        # Apply postprocessor if provided
        if postprocessor is not None:
            action = postprocessor(action.unsqueeze(0))
        else:
            action = action.unsqueeze(0)
        
        return action

    def reset(self):
        """Reset the manager state."""
        self.action_queue = []
        self.queue_index = 0
        self.inference_times = []
        self.total_inferences = 0
        self.policy.reset()

    def get_statistics(self) -> Dict[str, float]:
        """Get inference timing statistics."""
        if not self.inference_times:
            return {
                "mean_inference_time": 0.0,
                "min_inference_time": 0.0,
                "max_inference_time": 0.0,
                "total_inferences": 0,
            }
        
        return {
            "mean_inference_time": sum(self.inference_times) / len(self.inference_times),
            "min_inference_time": min(self.inference_times),
            "max_inference_time": max(self.inference_times),
            "total_inferences": self.total_inferences,
        }
