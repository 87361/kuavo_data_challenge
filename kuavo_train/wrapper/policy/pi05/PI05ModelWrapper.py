"""
PI05 Model Wrapper with LoRA support.

This module provides a custom wrapper around the PI05Pytorch model that adds:
- LoRA (Low-Rank Adaptation) fine-tuning support via the peft library
- Optional vision tower freezing
- Optional depth image branch support
"""

import logging
import torch
import torch.nn as nn
from typing import Optional

from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.policies.pi05.configuration_pi05 import PI05Config

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("peft library not available. LoRA support disabled.")


class CustomPI05ModelWrapper(PI05Pytorch):
    """Custom PI05 Model Wrapper with LoRA fine-tuning support.
    
    This wrapper extends PI05Pytorch to support:
    - LoRA fine-tuning for the PaliGemma language model and Gemma Expert
    - Optional freezing of the vision tower (SigLIP)
    - Optional depth image processing branch
    
    Args:
        config: PI05Config or CustomPI05ConfigWrapper with LoRA parameters
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Apply LoRA if enabled
        if getattr(config, 'use_lora', False):
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "peft library is required for LoRA. Install it with: pip install peft"
                )
            self._apply_lora(config)
        
        # Freeze vision tower if specified
        if getattr(config, 'freeze_vision_tower', True):
            self._freeze_vision_tower()
        
        # Initialize depth branch if enabled
        if getattr(config, 'use_depth', False):
            self._init_depth_branch(config)
    
    def _apply_lora(self, config):
        """Apply LoRA to the PaliGemma language model and Gemma Expert.
        
        LoRA is applied to the attention layers (q_proj, v_proj, k_proj, o_proj by default).
        This significantly reduces the number of trainable parameters while maintaining
        good fine-tuning performance.
        """
        lora_rank = getattr(config, 'lora_rank', 16)
        lora_alpha = getattr(config, 'lora_alpha', 32.0)
        lora_dropout = getattr(config, 'lora_dropout', 0.05)
        target_modules = getattr(config, 'lora_target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"])
        
        # Ensure target_modules is a plain Python list (convert from ListConfig if needed)
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        else:
            # Convert ListConfig or other iterables to plain list for JSON serialization
            target_modules = list(target_modules)
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to PaliGemma language model
        try:
            self.paligemma_with_expert.paligemma.language_model = get_peft_model(
                self.paligemma_with_expert.paligemma.language_model, 
                lora_config
            )
            logging.info(f"Applied LoRA to PaliGemma language model with rank={lora_rank}")
        except Exception as e:
            logging.warning(f"Failed to apply LoRA to PaliGemma language model: {e}")
        
        # Apply LoRA to Gemma Expert
        try:
            self.paligemma_with_expert.gemma_expert = get_peft_model(
                self.paligemma_with_expert.gemma_expert, 
                lora_config
            )
            logging.info(f"Applied LoRA to Gemma Expert with rank={lora_rank}")
        except Exception as e:
            logging.warning(f"Failed to apply LoRA to Gemma Expert: {e}")
        
        # Print trainable parameters info
        self._print_trainable_params()
    
    def _freeze_vision_tower(self):
        """Freeze the vision tower (SigLIP) parameters.
        
        This is recommended for fine-tuning as the vision encoder is usually
        well-pretrained and doesn't need updates for most tasks.
        """
        try:
            vision_tower = self.paligemma_with_expert.paligemma.vision_tower
            for param in vision_tower.parameters():
                param.requires_grad = False
            logging.info("Froze vision tower (SigLIP) parameters")
        except Exception as e:
            logging.warning(f"Failed to freeze vision tower: {e}")
    
    def _init_depth_branch(self, config):
        """Initialize depth image processing branch.
        
        If use_depth is enabled, this creates a separate backbone for processing
        depth images and fuses them with RGB features.
        """
        try:
            from torchvision.models import resnet18, resnet34, resnet50
            from torchvision.models._utils import IntermediateLayerGetter
            
            depth_backbone_name = getattr(config, 'depth_backbone', 'resnet18')
            
            # Select backbone
            if depth_backbone_name == 'resnet18':
                backbone = resnet18(weights=None)
            elif depth_backbone_name == 'resnet34':
                backbone = resnet34(weights=None)
            elif depth_backbone_name == 'resnet50':
                backbone = resnet50(weights=None)
            else:
                raise ValueError(f"Unknown depth backbone: {depth_backbone_name}")
            
            # Modify first conv layer for single-channel depth input
            backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Get intermediate features
            self.depth_backbone = IntermediateLayerGetter(
                backbone, return_layers={"layer4": "feature"}
            )
            
            # Projection layer to match embedding dimension
            # Get the embedding dimension from config
            from lerobot.policies.pi05.modeling_pi05 import get_gemma_config
            action_expert_config = get_gemma_config(config.action_expert_variant)
            embed_dim = action_expert_config.width
            
            # ResNet18/34 outputs 512 channels, ResNet50 outputs 2048
            if depth_backbone_name in ['resnet18', 'resnet34']:
                depth_feature_dim = 512
            else:
                depth_feature_dim = 2048
            
            self.depth_proj = nn.Linear(depth_feature_dim, embed_dim)
            
            logging.info(f"Initialized depth branch with {depth_backbone_name}")
            
        except Exception as e:
            logging.warning(f"Failed to initialize depth branch: {e}")
            self.depth_backbone = None
            self.depth_proj = None
    
    def _print_trainable_params(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0
        logging.info(
            f"Trainable params: {trainable_params:,} / {all_params:,} "
            f"({trainable_percent:.2f}%)"
        )
    
    def get_trainable_parameters(self):
        """Return only trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def merge_and_unload_lora(self):
        """Merge LoRA weights into base model and remove LoRA modules.
        
        This is useful for inference after training, as it removes the overhead
        of the LoRA adapters.
        """
        if not PEFT_AVAILABLE:
            logging.warning("peft not available, cannot merge LoRA")
            return
        
        try:
            if isinstance(self.paligemma_with_expert.paligemma.language_model, PeftModel):
                self.paligemma_with_expert.paligemma.language_model = \
                    self.paligemma_with_expert.paligemma.language_model.merge_and_unload()
                logging.info("Merged and unloaded LoRA for PaliGemma language model")
        except Exception as e:
            logging.warning(f"Failed to merge LoRA for PaliGemma: {e}")
        
        try:
            if isinstance(self.paligemma_with_expert.gemma_expert, PeftModel):
                self.paligemma_with_expert.gemma_expert = \
                    self.paligemma_with_expert.gemma_expert.merge_and_unload()
                logging.info("Merged and unloaded LoRA for Gemma Expert")
        except Exception as e:
            logging.warning(f"Failed to merge LoRA for Gemma Expert: {e}")
    
    def save_lora_weights(self, save_directory):
        """Save only the LoRA adapter weights.
        
        Args:
            save_directory: Path to save the LoRA weights
        """
        if not PEFT_AVAILABLE:
            logging.warning("peft not available, cannot save LoRA weights")
            return
        
        from pathlib import Path
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(self.paligemma_with_expert.paligemma.language_model, PeftModel):
                self.paligemma_with_expert.paligemma.language_model.save_pretrained(
                    save_path / "paligemma_lora"
                )
                logging.info(f"Saved PaliGemma LoRA weights to {save_path / 'paligemma_lora'}")
        except Exception as e:
            logging.warning(f"Failed to save PaliGemma LoRA weights: {e}")
        
        try:
            if isinstance(self.paligemma_with_expert.gemma_expert, PeftModel):
                self.paligemma_with_expert.gemma_expert.save_pretrained(
                    save_path / "gemma_expert_lora"
                )
                logging.info(f"Saved Gemma Expert LoRA weights to {save_path / 'gemma_expert_lora'}")
        except Exception as e:
            logging.warning(f"Failed to save Gemma Expert LoRA weights: {e}")
    
    def load_lora_weights(self, load_directory):
        """Load LoRA adapter weights.
        
        Args:
            load_directory: Path to load the LoRA weights from
        """
        if not PEFT_AVAILABLE:
            logging.warning("peft not available, cannot load LoRA weights")
            return
        
        from pathlib import Path
        from peft import PeftModel
        
        load_path = Path(load_directory)
        
        try:
            paligemma_lora_path = load_path / "paligemma_lora"
            if paligemma_lora_path.exists():
                self.paligemma_with_expert.paligemma.language_model = PeftModel.from_pretrained(
                    self.paligemma_with_expert.paligemma.language_model,
                    paligemma_lora_path
                )
                logging.info(f"Loaded PaliGemma LoRA weights from {paligemma_lora_path}")
        except Exception as e:
            logging.warning(f"Failed to load PaliGemma LoRA weights: {e}")
        
        try:
            gemma_expert_lora_path = load_path / "gemma_expert_lora"
            if gemma_expert_lora_path.exists():
                self.paligemma_with_expert.gemma_expert = PeftModel.from_pretrained(
                    self.paligemma_with_expert.gemma_expert,
                    gemma_expert_lora_path
                )
                logging.info(f"Loaded Gemma Expert LoRA weights from {gemma_expert_lora_path}")
        except Exception as e:
            logging.warning(f"Failed to load Gemma Expert LoRA weights: {e}")
