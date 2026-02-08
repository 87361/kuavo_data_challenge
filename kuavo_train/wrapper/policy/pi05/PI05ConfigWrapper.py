"""
PI0.5 配置包装器 - 继承 lerobot PI05Config，添加自定义扩展

@author 小华同学 ai
@created 2026-02-06
"""

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
from typing import TypeVar

T = TypeVar("T", bound="CustomPI05ConfigWrapper")


@PreTrainedConfig.register_subclass("custom_pi05")
@dataclass
class CustomPI05ConfigWrapper(PI05Config):
    """自定义 PI0.5 配置包装器，支持过滤深度特征和自定义参数。"""

    custom: Dict[str, Any] = field(default_factory=dict)

    # Task description for PI0.5 language input
    task_description: str = "Pick up the parcel and place it on the scale."

    # Whether to load from pretrained HuggingFace model
    pretrained_model_name: str = "lerobot/pi05_base"

    # LoRA configs
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def __post_init__(self):
        super().__post_init__()

        # Apply custom settings
        if isinstance(self.custom, (DictConfig, dict)):
            for k, v in self.custom.items():
                if not hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise ValueError(
                        f"Custom setting '{k}: {v}' conflicts with the parent base configuration. "
                        f"Remove it from 'custom' and modify in the parent configuration instead."
                    )
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
        """Only return RGB/VISUAL features, excluding depth."""
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type is FeatureType.VISUAL or ft.type is FeatureType.RGB
        }

    @property
    def depth_features(self) -> dict[str, PolicyFeature]:
        """Return depth features (for reference, PI0.5 does not use depth)."""
        return {
            key: ft
            for key, ft in self.input_features.items()
            if ft.type is FeatureType.DEPTH
        }

    def filter_depth_features(self):
        """Remove depth features from input_features since PI0.5 doesn't support depth."""
        depth_keys = [
            key for key, ft in self.input_features.items() if ft.type is FeatureType.DEPTH
        ]
        for key in depth_keys:
            del self.input_features[key]
        if depth_keys:
            print(f"PI0.5: Filtered out depth features: {depth_keys}")

    def validate_features(self) -> None:
        """Validate features after filtering depth."""
        self.filter_depth_features()
        super().validate_features()

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save config to directory."""
        cfg_copy = deepcopy(self)
        if isinstance(cfg_copy.custom, dict):
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
