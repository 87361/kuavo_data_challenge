import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
from lerobot.configs.policies import PolicyFeature
from typing import Any

import hydra
import json
from omegaconf import DictConfig, OmegaConf, ListConfig
from pathlib import Path
from functools import partial
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
from hydra.utils import instantiate
from diffusers.optimization import get_scheduler

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.random_utils import set_seed
from lerobot.policies.factory import make_pre_post_processors
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import CustomLeRobotDataset
from kuavo_train.utils.augmenter import crop_image, resize_image, DeterministicAugmenterColor
from kuavo_train.utils.feature_filter import (
    apply_feature_filter_to_metadata,
    feature_filter_from_cfg,
    make_feature_filter_step,
)
from kuavo_train.utils.utils import save_rng_state, load_rng_state
from lerobot.policies.act.modeling_act import ACTPolicy
from diffusers.optimization import get_scheduler
from utils.transforms import ImageTransforms, ImageTransformsConfig, ImageTransformConfig

from functools import partial
from contextlib import nullcontext
from lerobot.processor import ProcessorStep, NormalizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
# import ipdb


def build_augmenter(cfg):
    """Since operations such as cropping and resizing in LeRobot are implemented at the model level 
    rather than at the data level, we provide only RGB image augmentations on the data side here, 
    with support for customization. For more details, refer to configs/policy/diffusion_config.yaml. 
    To define custom transformations, please see utils.transforms.py."""

    img_tf_cfg = ImageTransformsConfig(
        enable=cfg.get("enable", False),
        max_num_transforms=cfg.get("max_num_transforms", 3),
        random_order=cfg.get("random_order", False),
        tfs={}
    )

    # deal tfs part
    if "tfs" in cfg:
        for name, tf_dict in cfg["tfs"].items():
            img_tf_cfg.tfs[name] = ImageTransformConfig(
                weight=tf_dict.get("weight", 1.0),
                type=tf_dict.get("type", "Identity"),
                kwargs=tf_dict.get("kwargs", {}),
            )
    return ImageTransforms(img_tf_cfg)


def build_delta_timestamps(dataset_metadata, policy_cfg):
    """Build delta timestamps for observations and actions."""
    obs_indices = getattr(policy_cfg, "observation_delta_indices", None)
    act_indices = getattr(policy_cfg, "action_delta_indices", None)
    if obs_indices is None and act_indices is None:
        return None

    delta_timestamps = {}
    for key in dataset_metadata.info["features"]:
        if "observation" in key and obs_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in obs_indices]
        elif "action" in key and act_indices is not None:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in act_indices]

    return delta_timestamps if delta_timestamps else None


def filter_delta_timestamps(delta_timestamps, filter_spec):
    if delta_timestamps is None or not filter_spec.get("enabled", False):
        return delta_timestamps
    allowed_keys = {
        filter_spec["state_key"],
        filter_spec["action_key"],
        *filter_spec["image_keys"],
    }
    return {key: value for key, value in delta_timestamps.items() if key in allowed_keys}


def build_optimizer_and_scheduler(policy, cfg, total_frames):
    """Return optimizer and scheduler."""
    optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
    # If `max_training_step` is specified, it takes precedence; 
    # otherwise, the value is automatically determined based on `max_epoch`.
    if cfg.training.max_training_step is None:
        updates_per_epoch = (total_frames // (cfg.training.batch_size * cfg.training.accumulation_steps)) + 1
        num_training_steps = cfg.training.max_epoch * updates_per_epoch
    else:
        num_training_steps = cfg.training.max_training_step
    lr_scheduler = policy.config.get_scheduler_preset()
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler.build(optimizer, num_training_steps)
    else:
        lr_scheduler = get_scheduler(
            name=cfg.training.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.scheduler_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # or you can set your optimizer and lr_scheduler here and replace it.
    return optimizer, lr_scheduler

def build_policy(name, policy_cfg):
    policy = {
        "diffusion": CustomDiffusionPolicyWrapper,
        "act": CustomACTPolicyWrapper,
    }[name](policy_cfg)
    return policy

def build_policy_config(cfg, input_features, output_features):
    def _normalize_feature_dict(d: Any) -> dict[str, PolicyFeature]:
        if isinstance(d, DictConfig):
            d = OmegaConf.to_container(d, resolve=True)
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict or DictConfig, got {type(d)}")

        return {
            k: PolicyFeature(**v) if isinstance(v, dict) and not isinstance(v, PolicyFeature) else v
            for k, v in d.items()
        }

    policy_cfg = instantiate(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
        device=cfg.training.device,
    )
                
    policy_cfg.input_features = _normalize_feature_dict(policy_cfg.input_features)
    policy_cfg.output_features = _normalize_feature_dict(policy_cfg.output_features)
    return policy_cfg

class AugmentationProcessorStep(ProcessorStep):
    def __init__(self, transform, cam_keys):
        super().__init__()
        self.transform = transform
        self.cam_keys = [k for k in cam_keys if "depth" not in k]  # list of keys in the transition dict to augment

    def __call__(self, transition):
        # Store the current transition (required by ProcessorStep)
        new_transition = transition.copy()

        # Apply transform to each camera key
        data_dict = new_transition.get(TransitionKey.OBSERVATION)
        if data_dict is not None:
            # new_data_dict = {
            #     k: self.transform(v) if k in self.cam_keys else v
            #     for k, v in data_dict.items()
            # }
            new_data_dict = {}
            for k, v in data_dict.items():
                
                if k in self.cam_keys:
                    # print(k)
                    new_data_dict[k] = self.transform(v)
                else:
                    new_data_dict[k] = v
            # print(new_data_dict['observation.images.head_cam_h'].device)
            new_transition[TransitionKey.OBSERVATION] = new_data_dict
            return new_transition
        else:
            return new_transition
        

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Device and dtype transformations do not alter the fundamental definition of the features (e.g., shape).

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features
    

def insert_before_normalizer(pipeline, new_step):
    """
    Insert a processor step before the first NormalizerProcessorStep.
    If no NormalizerProcessorStep is found, append at the end.
    """
    for i, step in enumerate(pipeline.steps):
        if isinstance(step, NormalizerProcessorStep):
            pipeline.steps.insert(i, new_step)
            print(f"Inserted {new_step.__class__.__name__} before NormalizerProcessorStep", {i})
            return new_step
    pipeline.steps.append(new_step)
    print(f"No NormalizerProcessorStep found, appended {new_step.__class__.__name__} at the end")
    return new_step

def remove_aug_step(pipeline, step_to_remove):
    """
    Remove the given step from the pipeline if it exists.
    """
    if step_to_remove in pipeline.steps:
        pipeline.steps.remove(step_to_remove)
        print(f"Removed {step_to_remove.__class__.__name__}")
    else:
        print(f"Step {step_to_remove.__class__.__name__} not found in pipeline")


def should_run_open_loop_eval(cfg, epoch):
    freq = cfg.training.get("open_loop_eval_freq_epoch", None)
    return freq is not None and freq > 0 and (epoch + 1) % freq == 0


def _act_open_loop_metrics(policy, batch):
    actions_hat = policy.predict_action_chunk(batch)
    target = batch["action"]
    diff = actions_hat - target
    action_is_pad = batch.get("action_is_pad", None)
    if action_is_pad is not None:
        valid_mask = (~action_is_pad).unsqueeze(-1).expand_as(diff).to(diff.dtype)
        denom = valid_mask.sum().clamp_min(1.0)
        l1_loss = (diff.abs() * valid_mask).sum() / denom
        mse_loss = (diff.pow(2) * valid_mask).sum() / denom
    else:
        l1_loss = diff.abs().mean()
        mse_loss = diff.pow(2).mean()
    return {"loss": l1_loss, "l1_loss": l1_loss, "mse_loss": mse_loss}


def _open_loop_metrics(policy, batch, cfg):
    if str(cfg.get("policy_name", "")) == "act":
        return _act_open_loop_metrics(policy, batch)
    loss, _ = policy.forward(batch)
    return {"loss": loss}


def run_open_loop_eval(policy, dataloader, preprocessor, cfg, make_autocast, amp_enabled, writer, output_directory, epoch, steps):
    policy.eval()
    metric_totals = {}
    batch_count = 0
    max_batches = cfg.training.get("open_loop_eval_max_batches", None)
    eval_bar = tqdm(dataloader, desc=f"Open-loop eval epoch {epoch + 1}", leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_bar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = preprocessor(batch)
            with make_autocast(amp_enabled):
                metrics = _open_loop_metrics(policy, batch, cfg)
            for name, value in metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + value.detach().item()
            batch_count += 1
            eval_bar.set_postfix(loss=f"{metrics['loss'].detach().item():.3f}")

    mean_metrics = {
        name: total / batch_count
        for name, total in metric_totals.items()
    } if batch_count > 0 else {"loss": float("nan")}
    mean_loss = mean_metrics["loss"]
    writer.add_scalar("open_loop/loss", mean_loss, steps)
    for name, value in mean_metrics.items():
        if name != "loss":
            writer.add_scalar(f"open_loop/{name}", value, steps)
    metrics_path = output_directory / "open_loop_eval_metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": epoch + 1, "steps": steps, **mean_metrics}) + "\n")
    metric_summary = ", ".join(f"{name}={value:.6f}" for name, value in mean_metrics.items())
    print(f"Open-loop eval epoch {epoch + 1}: {metric_summary}")
    policy.train()
    return mean_loss

@hydra.main(config_path="../configs/policy/", config_name="diffusion_config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.training.seed)

    # Setup output directory
    output_directory = Path(cfg.training.output_directory) / f"run_{cfg.timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory))

    device = torch.device(cfg.training.device)

    # Dataset metadata and features
    dataset_metadata = LeRobotDatasetMetadata(cfg.repoid, root=cfg.root)
    print("Camera_keys:", dataset_metadata.camera_keys)
    print("Original dataset features:", dataset_metadata.features)

    features = dataset_to_policy_features(dataset_metadata.features)
    input_features = {k: ft for k, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    filter_spec = feature_filter_from_cfg(cfg)
    input_features, output_features, dataset_stats = apply_feature_filter_to_metadata(
        input_features,
        output_features,
        dataset_metadata.stats,
        filter_spec,
    )

    print(f"Input features: {input_features}")
    print(f"Output features: {output_features}")
    print(f"Feature filter: {filter_spec}")

    # instantiate the policy
    policy_cfg = build_policy_config(cfg, input_features, output_features)
    print("policy_cfg", policy_cfg)

    # Build policy
    policy = build_policy(cfg.policy_name, policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=dataset_stats)
    filter_step = make_feature_filter_step(filter_spec)
    if filter_step is not None:
        insert_before_normalizer(preprocessor, filter_step)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    eval_preprocessor = deepcopy(preprocessor)
    optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"])
    
    # Initialize AMP GradScaler if use_amp is True
    amp_requested = bool(getattr(cfg.policy, "use_amp", False))
    amp_enabled = amp_requested and device.type == "cuda"

    # autocast context (cuda, or no-op when disabled/non-cuda)
    has_torch_autocast = hasattr(torch, "autocast")
    def make_autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        if device.type == "cuda":
            if has_torch_autocast:
                return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)  # noqa
            else:
                from torch.cuda.amp import autocast as cuda_autocast  # noqa
                return cuda_autocast()
        # Fallback: disable on non-cuda to avoid dtype surprises
        return nullcontext()

    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(device=device.type, enabled=amp_enabled)
    # print("scaler", device.type, make_autocast(amp_enabled))
    # Initialize training state variables
    start_epoch = 0
    steps = 0
    best_loss = float('inf')

    # ===== Resume logic (perfect resume for AMP & RNG) =====
    
    if cfg.training.resume and cfg.training.resume_timestamp:
        resume_path = Path(cfg.training.output_directory) / cfg.training.resume_timestamp
        print("Resuming from:", resume_path)
        try:
            # Load RNG state
            load_rng_state(resume_path / "rng_state.pth")
            
            # Load policy
            policy = policy.from_pretrained(resume_path, strict=True)
            preprocessor = preprocessor.from_pretrained(resume_path,config_filename="policy_preprocessor.json")

            """ Warning: using `from_pretrained` creates a new policy instance, 
            so the optimizer must be reinitialized here! """
            # print("load policy done ! ")
            optimizer, lr_scheduler = build_optimizer_and_scheduler(policy, cfg, dataset_metadata.info["total_frames"])
            
            # Load optimizer, scheduler, scaler and training state
            checkpoint = torch.load(resume_path / "learning_state.pth", map_location=device)
            optimizer.load_state_dict(checkpoint["optimizer"])

            if "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            
            if "scaler" in checkpoint and amp_enabled:
                scaler.load_state_dict(checkpoint["scaler"])
            
            if "steps" in checkpoint:
                steps = checkpoint["steps"]
            
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"]
            
            if "best_loss" in checkpoint:
                best_loss = checkpoint["best_loss"]
            
            # Copy and load log_event
            for file in resume_path.glob("events.*"):
                shutil.copy(file, output_directory)
                
            print(f"Resumed training from epoch {start_epoch}, step {steps}")
        except Exception as e:
            print("Failed to load checkpoint:", e)
            return
    else:
        print("Training from scratch!")

    policy.train().to(device)
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Using AMP: {amp_enabled}")
    # Build dataset and dataloader
    delta_timestamps = filter_delta_timestamps(build_delta_timestamps(dataset_metadata, policy_cfg), filter_spec)

    image_transforms = build_augmenter(cfg.training.RGB_Augmenter)
    dataset = LeRobotDataset(
        cfg.repoid,
        delta_timestamps=delta_timestamps,
        root=cfg.root,
        image_transforms=None,
    )
    # Training loop
    aug_step = insert_before_normalizer(preprocessor, AugmentationProcessorStep(image_transforms, dataset.meta.camera_keys))  # just for training
    
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    
    stop_training = False
    max_training_step = cfg.training.max_training_step
    for epoch in range(start_epoch, cfg.training.max_epoch):
        dataloader = DataLoader(
            dataset,
            num_workers=cfg.training.num_workers,
            batch_size=cfg.training.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=(device.type != "cpu"),
            drop_last=cfg.training.drop_last,
            prefetch_factor=2 if cfg.training.num_workers > 0 else None,
        )

        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.max_epoch}")

        
        total_loss = 0.0
        for batch in epoch_bar:
            if max_training_step is not None and steps >= max_training_step:
                stop_training = True
                break
            batch = preprocessor(batch)  # will normalize and put batch to device
            with make_autocast(amp_enabled):
                loss, _ = policy.forward(batch)
            # Scale loss and backward with AMP if enabled
            scaled_loss = loss / cfg.training.accumulation_steps
            
            if amp_enabled:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if steps % cfg.training.accumulation_steps == 0:
                if amp_enabled:
                    # Optionally unscale and clip gradients here if you use clipping
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if steps % cfg.training.log_freq == 0:
                writer.add_scalar("train/loss", scaled_loss.item(), steps)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], steps)
                epoch_bar.set_postfix(loss=f"{scaled_loss.item():.3f}", step=steps, lr=lr_scheduler.get_last_lr()[0])

            steps += 1
            total_loss += scaled_loss.item()
            if max_training_step is not None and steps >= max_training_step:
                stop_training = True
                break
        
        # Update best loss
        if total_loss < best_loss:
            best_loss = total_loss
            # Save best model
            policy.save_pretrained(output_directory / "epochbest")
        # Save checkpoint every N epochs
        if (epoch + 1) % cfg.training.save_freq_epoch == 0:
            policy.save_pretrained(output_directory / f"epoch{epoch+1}")
            # preprocessor.save_pretrained(output_directory)

        # Save last checkpoint (includes AMP scaler & progress for perfect resume)
        # Save last checkpoint
        policy.save_pretrained(output_directory)

        # Save training state including optimizer, scheduler, scaler, and step/epoch info
        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "steps": steps,
            "epoch": epoch + 1,
            "best_loss": best_loss
        }
        torch.save(checkpoint, output_directory / "learning_state.pth")
        save_rng_state(output_directory / "rng_state.pth")

        if should_run_open_loop_eval(cfg, epoch):
            run_open_loop_eval(
                policy,
                dataloader,
                eval_preprocessor,
                cfg,
                make_autocast,
                amp_enabled,
                writer,
                output_directory,
                epoch,
                steps,
            )

        if stop_training:
            print(f"Reached training.max_training_step={max_training_step}; stopping.")
            break

    writer.close()


if __name__ == "__main__":
    main()
