from lerobot.policies.act.modeling_act import ACTPolicy
from kuavo_train.utils.augmenter import (crop_image,
                                        resize_image)
from torch import Tensor, nn
import torch
from collections import deque
from lerobot.utils.constants  import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from kuavo_train.wrapper.policy.act.ACTConfigWrapper import CustomACTConfigWrapper

from kuavo_train.wrapper.policy.act.ACTModelWrapper import CustomACTModelWrapper
import os, builtins
from pathlib import Path
from typing import TypeVar
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
import torchvision
import torchvision.transforms.functional
import torch.nn.functional as F

T = TypeVar("T", bound="CustomACTPolicyWrapper")
OBS_DEPTH = "observation.depth"

class CustomACTPolicyWrapper(ACTPolicy):
    def __init__(self,         
                 config: CustomACTConfigWrapper,
    ):
        super().__init__(config)
        self.model = CustomACTModelWrapper(config)


    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        if self.config.use_depth and self.config.depth_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_DEPTH] = [batch[key].mean(dim=-3, keepdim=True) for key in self.config.depth_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation.

        When use_world_model is enabled and we are training, this also computes:
          - L_dyn:  dynamics loss  (predict next proprioceptive state from z_t, a_t)
          - L_cf:   counterfactual consistency loss (swap z sub-block, enforce invariance)
        Total: L = L_IL + λ_dyn * L_dyn + λ_cf * L_cf
        """
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        
        if self.config.use_depth and self.config.depth_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_DEPTH] = [batch[key].mean(dim=-3, keepdim=True) for key in self.config.depth_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # ===== Original IL Loss (completely unchanged) =====
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
            # each dimension independently, we sum over the latent dimension to get the total
            # KL-divergence per batch element, then take the mean over the batch.
            # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        # ===== World Model Losses (only during training, fully gated by config) =====
        if getattr(self.config, 'use_world_model', False) and self.training:
            encoder_out = self.model._last_encoder_out  # (S, B, 512)
            # z_t = latent token position after self-attention.
            # DETACH: world model gradients do NOT flow back through the encoder.
            z_t = encoder_out[0].detach()               # (B, 512)
            a_t = batch[ACTION][:, 0, :]                # (B, 16)  first action step

            # --- Dynamics loss: predict next proprioceptive state ---
            l_dyn = torch.tensor(0.0, device=z_t.device)
            if 'next_observation.state' in batch:
                s_next_target = batch['next_observation.state']              # (B, 16)
                s_next_pred = self.model.dynamics_head(z_t, a_t)             # (B, 16)
                l_dyn = F.mse_loss(s_next_pred, s_next_target)

            # --- Counterfactual consistency loss ---
            from kuavo_train.wrapper.policy.act.WorldModelHead import compute_counterfactual_loss
            l_cf = compute_counterfactual_loss(
                dynamics_head=self.model.dynamics_head,
                z_t=z_t,
                a_t=a_t,
                n_z_blocks=getattr(self.config, 'wm_n_z_blocks', 4),
                swap_block_idx=getattr(self.config, 'wm_swap_block_idx', 3),
                cf_mask=None,  # None → all state dims must be invariant to scene swap
            )

            # --- Warmup scheduling (coefficient set by training loop) ---
            warmup_coeff = getattr(self, '_wm_warmup_coeff', 1.0)
            lambda_dyn = self.config.wm_lambda_dyn * warmup_coeff
            lambda_cf  = self.config.wm_lambda_cf  * warmup_coeff

            loss_dict['dyn_loss'] = l_dyn.item()
            loss_dict['cf_loss'] = l_cf.item()
            loss_dict['wm_warmup'] = warmup_coeff

            loss = loss + lambda_dyn * l_dyn + lambda_cf * l_cf
        # ===== End World Model =====

        return loss, loss_dict


    
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: CustomACTConfigWrapper | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = CustomACTConfigWrapper.from_pretrained(
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
        model_id = str(pretrained_name_or_path)
        # print(config)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if p.requires_grad
                    and not (n.startswith("model.backbone") or n.startswith("model.depth_backbone"))
                ],
                "lr": self.config.optimizer_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if p.requires_grad and (n.startswith("model.backbone") or n.startswith("model.depth_backbone"))
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]