"""
PI0.5 策略包装器 - 继承 lerobot PI05Policy，适配 kuavo 训练框架

@author 小华同学 ai
@created 2026-02-06
"""

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from kuavo_train.wrapper.policy.pi05.PI05ConfigWrapper import CustomPI05ConfigWrapper
from torch import Tensor
import torch
import builtins
import os
from pathlib import Path
from typing import TypeVar
from lerobot.configs.policies import PreTrainedConfig

T = TypeVar("T", bound="CustomPI05PolicyWrapper")


class CustomPI05PolicyWrapper(PI05Policy):
    """自定义 PI0.5 策略包装器。
    
    主要功能:
    - 过滤深度特征，仅使用 RGB 图像
    - 支持从 HuggingFace 加载预训练权重
    - 适配 kuavo 训练框架的 forward/loss 接口
    """

    def __init__(self, config: CustomPI05ConfigWrapper):
        super().__init__(config)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.
        
        Delegates to the parent PI05Policy.forward() which handles:
        - Image preprocessing
        - Language token processing
        - Action padding and loss computation
        """
        return super().forward(batch)

    def get_optim_params(self):
        """Return model parameters for optimizer."""
        return self.parameters()
