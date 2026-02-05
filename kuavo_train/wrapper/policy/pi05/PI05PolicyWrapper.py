"""
PI05 Policy Wrapper with LoRA support.

This module provides a custom wrapper around the PI05Policy that integrates:
- LoRA fine-tuning via CustomPI05ModelWrapper
- Custom from_pretrained loading with LoRA support
- Optimized parameter selection for LoRA training
"""

import builtins
import logging
import re
from collections import deque
from pathlib import Path
from typing import TypeVar

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, pad_vector
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from kuavo_train.wrapper.policy.pi05.PI05ConfigWrapper import CustomPI05ConfigWrapper
from kuavo_train.wrapper.policy.pi05.PI05ModelWrapper import CustomPI05ModelWrapper

T = TypeVar("T", bound="CustomPI05PolicyWrapper")


class CustomPI05PolicyWrapper(PI05Policy):
    """Custom PI05 Policy Wrapper with LoRA fine-tuning support.
    
    This wrapper extends PI05Policy to support:
    - LoRA fine-tuning via CustomPI05ModelWrapper
    - Custom parameter selection for optimizer
    - LoRA weight saving/loading
    
    Args:
        config: CustomPI05ConfigWrapper with LoRA and depth parameters
    """
    
    config_class = CustomPI05ConfigWrapper
    name = "custom_pi05"
    
    def __init__(self, config: CustomPI05ConfigWrapper):
        """Initialize the policy with custom model wrapper.
        
        Args:
            config: Configuration with LoRA and custom parameters
        """
        # Call grandparent init to avoid double model initialization
        # PreTrainedPolicy.__init__
        super(PI05Policy, self).__init__(config)
        config.validate_features()
        self.config = config
        
        # Initialize with custom model wrapper that supports LoRA
        self.model = CustomPI05ModelWrapper(config)
        
        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if config.device:
            self.model.to(config.device)
        
        self.reset()
    
    def get_optim_params(self):
        """Return parameters for optimizer.
        
        If LoRA is enabled, only returns trainable parameters (LoRA + projection layers).
        Otherwise returns all model parameters.
        """
        if getattr(self.config, 'use_lora', False):
            # Return only trainable parameters (LoRA adapters + unfrozen layers)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            logging.info(f"Returning {len(trainable_params)} trainable parameters for LoRA training")
            return trainable_params
        return self.model.parameters()
    
    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
    
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load a pretrained policy with optional LoRA weights.
        
        This method handles:
        - Loading base model weights
        - Loading LoRA adapter weights if present
        - Key remapping for compatibility
        """
        print(
            "Loading CustomPI05PolicyWrapper - a custom wrapper for PI05 with LoRA support.\n"
            "Based on the OpenPI implementation: https://github.com/Physical-Intelligence/openpi"
        )
        
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")
        
        pretrained_path = Path(pretrained_name_or_path)
        
        # Load config if not provided
        if config is None:
            try:
                config = PreTrainedConfig.from_pretrained(
                    pretrained_name_or_path=pretrained_name_or_path,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    revision=revision,
                    **kwargs,
                )
            except Exception as e:
                logging.warning(f"Could not load config from {pretrained_name_or_path}: {e}")
                logging.info("Creating default CustomPI05ConfigWrapper")
                config = CustomPI05ConfigWrapper()
        
        # Initialize model without loading weights
        model = cls(config, **kwargs)
        
        # Try to load state dict
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            
            # Try safetensors first, then pytorch_model.bin
            state_dict = None
            
            # Check for local files
            if pretrained_path.is_dir():
                safetensors_path = pretrained_path / "model.safetensors"
                pytorch_path = pretrained_path / "pytorch_model.bin"
                
                if safetensors_path.exists():
                    from safetensors.torch import load_file
                    state_dict = load_file(str(safetensors_path))
                    print("✓ Loaded state dict from model.safetensors")
                elif pytorch_path.exists():
                    state_dict = torch.load(str(pytorch_path), map_location="cpu")
                    print("✓ Loaded state dict from pytorch_model.bin")
            
            # Try HuggingFace Hub if not local
            if state_dict is None:
                try:
                    from transformers.utils import cached_file
                    resolved_file = cached_file(
                        pretrained_name_or_path,
                        "model.safetensors",
                        cache_dir=kwargs.get("cache_dir"),
                        force_download=kwargs.get("force_download", False),
                        resume_download=kwargs.get("resume_download"),
                        proxies=kwargs.get("proxies"),
                        use_auth_token=kwargs.get("use_auth_token"),
                        revision=kwargs.get("revision"),
                        local_files_only=kwargs.get("local_files_only", False),
                    )
                    from safetensors.torch import load_file
                    state_dict = load_file(resolved_file)
                    print("✓ Loaded state dict from HuggingFace Hub")
                except Exception as e:
                    print(f"Could not load from HuggingFace Hub: {e}")
            
            if state_dict is not None:
                # Fix state dict keys
                fixed_state_dict = model._fix_pytorch_state_dict_keys(state_dict, config)
                
                # Add "model." prefix if needed
                remapped_state_dict = {}
                remap_count = 0
                
                for key, value in fixed_state_dict.items():
                    if not key.startswith("model."):
                        new_key = f"model.{key}"
                        remapped_state_dict[new_key] = value
                        remap_count += 1
                    else:
                        remapped_state_dict[key] = value
                
                if remap_count > 0:
                    print(f"Remapped {remap_count} state dict keys")
                
                # Load state dict
                missing_keys, unexpected_keys = model.load_state_dict(
                    remapped_state_dict, strict=strict
                )
                
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    if len(missing_keys) > 5:
                        print(f"  ... and {len(missing_keys) - 5} more")
                
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    if len(unexpected_keys) > 5:
                        print(f"  ... and {len(unexpected_keys) - 5} more")
                
                if not missing_keys and not unexpected_keys:
                    print("✓ All keys loaded successfully!")
            else:
                print("Warning: No weights loaded, using random initialization")
                
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
        
        # Try to load LoRA weights if present
        if pretrained_path.is_dir():
            lora_path = pretrained_path / "lora"
            if lora_path.exists():
                try:
                    model.model.load_lora_weights(lora_path)
                    print("✓ Loaded LoRA weights")
                except Exception as e:
                    print(f"Warning: Could not load LoRA weights: {e}")
        
        return model
    
    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        """Fix state dict keys to match current model architecture."""
        fixed_state_dict = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # Handle layer norm structure changes for gemma expert
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                ) if hasattr(self.model.paligemma_with_expert.gemma_expert, 'config') else False
                if expert_uses_adarms:
                    logging.debug(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue
            
            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                ) if hasattr(self.model.paligemma_with_expert.gemma_expert, 'config') else False
                if expert_uses_adarms:
                    logging.debug(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue
            
            # Handle MLP naming changes for pi05
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            
            # Skip state_proj which doesn't exist in pi05
            if key.startswith("state_proj."):
                logging.debug(f"Skipping state_proj key in pi05 mode: {key}")
                continue
            
            fixed_state_dict[new_key] = value
        
        return fixed_state_dict
    
    def save_pretrained(self, save_directory: str | Path, **kwargs):
        """Save the policy to a directory.
        
        This saves:
        - Model configuration
        - Model weights (full or with LoRA adapters)
        - LoRA weights separately if using LoRA
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config._save_pretrained(save_path)
        
        # Get state dict
        state_dict = self.state_dict()
        
        # Handle shared tensors (e.g., lm_head.weight and embed_tokens.weight in PaliGemma)
        # Clone tensors that share memory to avoid safetensors error
        data_ptrs = {}
        for key, tensor in state_dict.items():
            ptr = tensor.data_ptr()
            if ptr in data_ptrs:
                # This tensor shares memory with another, clone it
                state_dict[key] = tensor.clone()
            else:
                data_ptrs[ptr] = key
        
        # Save model weights using safetensors
        from safetensors.torch import save_file
        save_file(state_dict, save_path / "model.safetensors")
        print(f"✓ Saved model to {save_path / 'model.safetensors'}")
        
        # Save LoRA weights separately if using LoRA
        if getattr(self.config, 'use_lora', False):
            lora_path = save_path / "lora"
            self.model.save_lora_weights(lora_path)
    
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.
        
        Args:
            batch: Dictionary containing:
                - Image features (observation.images.*)
                - Language tokens and masks
                - Actions
        
        Returns:
            Tuple of (loss, loss_dict)
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        
        actions = self.prepare_action(batch)
        
        # Compute loss
        losses = self.model.forward(images, img_masks, tokens, masks, actions)
        
        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        
        loss = losses.mean()
        
        loss_dict = {
            "loss": loss.item(),
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }
        
        return loss, loss_dict
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.
        
        Uses action queue for temporal consistency when n_action_steps > 1.
        """
        self.eval()
        
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        
        return self._action_queue.popleft()
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        
        # Sample actions using the model
        actions = self.model.sample_actions(images, img_masks, tokens, masks)
        
        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        
        return actions
    
    def prepare_action(self, batch):
        """Pad action to max_action_dim."""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
