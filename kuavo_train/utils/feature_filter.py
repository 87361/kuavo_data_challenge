from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep
from lerobot.processor.core import EnvTransition, TransitionKey


OBS_STATE = "observation.state"
ACTION = "action"
OBS_IMAGE_PREFIX = "observation.images."


def _to_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_tuple(value: Any) -> tuple:
    value = _to_plain(value)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _normalize_image_key(key: str) -> str:
    if key.startswith("observation."):
        return key
    return f"{OBS_IMAGE_PREFIX}{key}"


def feature_filter_from_cfg(cfg: Any) -> dict[str, Any]:
    filter_cfg = _to_plain(cfg.get("feature_filter", {})) if hasattr(cfg, "get") else {}
    enabled = bool(filter_cfg.get("enabled", False))
    image_keys = tuple(_normalize_image_key(str(k)) for k in _as_tuple(filter_cfg.get("image_keys")))
    state_indices = tuple(int(i) for i in _as_tuple(filter_cfg.get("state_indices")))
    action_indices = tuple(int(i) for i in _as_tuple(filter_cfg.get("action_indices")))
    state_key = str(filter_cfg.get("state_key", OBS_STATE))
    action_key = str(filter_cfg.get("action_key", ACTION))

    if enabled:
        if not image_keys:
            raise ValueError("feature_filter.image_keys must be set when feature_filter.enabled=true")
        if not state_indices:
            raise ValueError("feature_filter.state_indices must be set when feature_filter.enabled=true")
        if not action_indices:
            raise ValueError("feature_filter.action_indices must be set when feature_filter.enabled=true")

    return {
        "enabled": enabled,
        "image_keys": image_keys,
        "state_indices": state_indices,
        "action_indices": action_indices,
        "state_key": state_key,
        "action_key": action_key,
    }


def _slice_last_dim(value: Any, indices: tuple[int, ...]) -> Any:
    if not indices:
        return value
    if isinstance(value, torch.Tensor):
        index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(dim=value.ndim - 1, index=index)
    if isinstance(value, np.ndarray):
        return np.take(value, indices, axis=value.ndim - 1)
    if isinstance(value, (list, tuple)):
        sliced = [value[i] for i in indices]
        return type(value)(sliced) if isinstance(value, tuple) else sliced
    return value


def _slice_stat_value(value: Any, indices: tuple[int, ...]) -> Any:
    if not indices:
        return deepcopy(value)
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] > max(indices):
            index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
            return value.index_select(0, index)
        return value.clone()
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] > max(indices):
            return np.take(value, indices, axis=0)
        return value.copy()
    if isinstance(value, list):
        if len(value) > max(indices):
            return [deepcopy(value[i]) for i in indices]
        return deepcopy(value)
    return deepcopy(value)


def _slice_stats_entry(entry: dict[str, Any], indices: tuple[int, ...]) -> dict[str, Any]:
    return {name: _slice_stat_value(value, indices) for name, value in entry.items()}


def _feature_with_vector_len(feature: PolicyFeature, length: int) -> PolicyFeature:
    return PolicyFeature(type=feature.type, shape=(length,))


def apply_feature_filter_to_metadata(
    input_features: dict[str, PolicyFeature],
    output_features: dict[str, PolicyFeature],
    dataset_stats: dict[str, dict[str, Any]],
    filter_spec: dict[str, Any],
) -> tuple[dict[str, PolicyFeature], dict[str, PolicyFeature], dict[str, dict[str, Any]]]:
    if not filter_spec.get("enabled", False):
        return input_features, output_features, dataset_stats

    image_keys = set(filter_spec["image_keys"])
    state_key = filter_spec["state_key"]
    action_key = filter_spec["action_key"]
    state_indices = tuple(filter_spec["state_indices"])
    action_indices = tuple(filter_spec["action_indices"])

    missing_images = sorted(image_keys.difference(input_features))
    if missing_images:
        raise KeyError(f"Selected image keys are not in dataset features: {missing_images}")
    if state_key not in input_features:
        raise KeyError(f"Selected state key is not in input features: {state_key}")
    if action_key not in output_features:
        raise KeyError(f"Selected action key is not in output features: {action_key}")

    filtered_input = {
        key: feature
        for key, feature in input_features.items()
        if key in image_keys
    }
    filtered_input[state_key] = _feature_with_vector_len(input_features[state_key], len(state_indices))
    filtered_input = {
        key: filtered_input[key]
        for key in [state_key, *filter_spec["image_keys"]]
        if key in filtered_input
    }

    filtered_output = {
        action_key: _feature_with_vector_len(output_features[action_key], len(action_indices))
    }

    filtered_stats = deepcopy(dataset_stats)
    if state_key in filtered_stats:
        filtered_stats[state_key] = _slice_stats_entry(filtered_stats[state_key], state_indices)
    if action_key in filtered_stats:
        filtered_stats[action_key] = _slice_stats_entry(filtered_stats[action_key], action_indices)

    return filtered_input, filtered_output, filtered_stats


class FeatureFilterProcessorStep(ProcessorStep):
    def __init__(
        self,
        *,
        enabled: bool = False,
        image_keys: list[str] | tuple[str, ...] | None = None,
        state_indices: list[int] | tuple[int, ...] | None = None,
        action_indices: list[int] | tuple[int, ...] | None = None,
        state_key: str = OBS_STATE,
        action_key: str = ACTION,
    ):
        super().__init__()
        self.enabled = enabled
        self.image_keys = tuple(_normalize_image_key(str(k)) for k in (image_keys or ()))
        self.state_indices = tuple(int(i) for i in (state_indices or ()))
        self.action_indices = tuple(int(i) for i in (action_indices or ()))
        self.state_key = state_key
        self.action_key = action_key

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        new_transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_observation = {}
            if self.state_key in observation:
                new_observation[self.state_key] = _slice_last_dim(observation[self.state_key], self.state_indices)
            else:
                raise KeyError(f"Missing selected state key in batch: {self.state_key}")

            for key in self.image_keys:
                if key not in observation:
                    raise KeyError(f"Missing selected image key in batch: {key}")
                new_observation[key] = observation[key]

            new_transition[TransitionKey.OBSERVATION] = new_observation

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            new_transition[TransitionKey.ACTION] = _slice_last_dim(action, self.action_indices)

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "image_keys": list(self.image_keys),
            "state_indices": list(self.state_indices),
            "action_indices": list(self.action_indices),
            "state_key": self.state_key,
            "action_key": self.action_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if not self.enabled:
            return features

        transformed = deepcopy(features)
        obs_features = transformed.get(PipelineFeatureType.OBSERVATION, {})
        action_features = transformed.get(PipelineFeatureType.ACTION, {})

        if self.state_key in obs_features:
            obs_features[self.state_key] = _feature_with_vector_len(obs_features[self.state_key], len(self.state_indices))
        obs_features = {
            key: obs_features[key]
            for key in [self.state_key, *self.image_keys]
            if key in obs_features
        }
        transformed[PipelineFeatureType.OBSERVATION] = obs_features

        if self.action_key in action_features:
            action_features = {
                self.action_key: _feature_with_vector_len(action_features[self.action_key], len(self.action_indices))
            }
            transformed[PipelineFeatureType.ACTION] = action_features

        return transformed


def make_feature_filter_step(filter_spec: dict[str, Any]) -> FeatureFilterProcessorStep | None:
    if not filter_spec.get("enabled", False):
        return None
    return FeatureFilterProcessorStep(**filter_spec)
