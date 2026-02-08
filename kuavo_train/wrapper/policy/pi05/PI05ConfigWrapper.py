from typing import Any, Dict
from dataclasses import dataclass, fields, field
import copy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from omegaconf import DictConfig, OmegaConf, ListConfig
from copy import deepcopy
from pathlib import Path
import draccus
from huggingface_hub.constants import CONFIG_NAME
import os
import builtins
import json
import tempfile
from typing import TypeVar
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError

T = TypeVar("T", bound="CustomPI05ConfigWrapper")


@PreTrainedConfig.register_subclass("custom_pi05")
@dataclass
class CustomPI05ConfigWrapper(PI05Config):
    """Custom PI05 Configuration Wrapper with LoRA and depth image support.
    
    This wrapper extends PI05Config to support:
    - LoRA fine-tuning configuration
    - Optional depth image input
    - Custom parameters via the 'custom' dictionary
    """
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        
        # Default normalization mapping for Pi05
        default_map = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }

        # Merge and update the normalization_mapping
        merged = copy.deepcopy(default_map)
        merged.update(self.normalization_mapping)
        self.normalization_mapping = merged
        
        # Make custom settings in main config for better access
        if isinstance(self.custom, DictConfig) or isinstance(self.custom, dict):
            for k, v in self.custom.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise ValueError(
                        f"Custom setting '{k}: {v}' conflicts with the parent base configuration. "
                        "Remove it from 'custom' and modify in the parent configuration instead."
                    )
        
        # Set default LoRA parameters if not specified (Updated to match VLASH)
        if not hasattr(self, 'use_lora'):
            self.use_lora = False
        if not hasattr(self, 'lora_rank'):
            self.lora_rank = 16
        if not hasattr(self, 'lora_alpha'):
            self.lora_alpha = 16  # Changed to 16 to match VLASH
        if not hasattr(self, 'lora_dropout'):
            self.lora_dropout = 0.0  # Changed to 0 to match VLASH
        if not hasattr(self, 'lora_target_modules'):
            # Expanded to include MLP layers (matching VLASH)
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj",      # MLP layers
                "out_proj", "fc1", "fc2",                 # Other linear layers
            ]
        if not hasattr(self, 'lora_modules_to_save'):
            # PI05-specific layers that should be fully trainable (critical for action generation)
            # Note: state_proj/state_mlp_in/state_mlp_out don't exist in PI05, removed
            self.lora_modules_to_save = [
                "action_in_proj", "action_out_proj",      # Action projection layers
                "time_mlp_in", "time_mlp_out",            # Time MLP layers
            ]
        if not hasattr(self, 'freeze_vision_tower'):
            self.freeze_vision_tower = True
        if not hasattr(self, 'use_depth'):
            self.use_depth = False
        if not hasattr(self, 'depth_backbone'):
            self.depth_backbone = "resnet18"
        if not hasattr(self, 'default_task'):
            self.default_task = "manipulation task"
            
        self._convert_omegaconf_fields()
    
    def _convert_omegaconf_fields(self):
        """Convert OmegaConf fields to native Python types."""
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (ListConfig, DictConfig)):
                converted = OmegaConf.to_container(val, resolve=True)
                setattr(self, f.name, converted)

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        """Get RGB/VISUAL type features."""
        return {
            key: ft for key, ft in self.input_features.items() 
            if (ft.type is FeatureType.RGB) or (ft.type is FeatureType.VISUAL)
        }
    
    @property
    def depth_features(self) -> dict[str, PolicyFeature]:
        """Get DEPTH type features."""
        return {
            key: ft for key, ft in self.input_features.items() 
            if ft.type is FeatureType.DEPTH
        }

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        # Call parent validation
        super().validate_features()
        
        # Additional validation for depth features if enabled
        if getattr(self, 'use_depth', False) and len(self.depth_features) == 0:
            print("Warning: use_depth is True but no depth features found in input_features")
        
        # Check image shape consistency
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, "
                        "but we expect all image shapes to match."
                    )
            
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save configuration to directory."""
        cfg_copy = deepcopy(self)
        # Remove custom attributes that were promoted to class level
        if isinstance(cfg_copy.custom, dict):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        elif hasattr(cfg_copy, "custom") and hasattr(cfg_copy.custom, "keys"):
            for k in list(cfg_copy.custom.keys()):
                if hasattr(cfg_copy, k):
                    delattr(cfg_copy, k)
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(cfg_copy, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs,
    ) -> T:
        """Load configuration from pretrained path."""
        parent_cls = PreTrainedConfig 
        return parent_cls.from_pretrained(
            pretrained_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            **policy_kwargs,
        )
