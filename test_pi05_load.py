"""简单测试脚本：验证 Pi05 模型能否正常加载"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from pathlib import Path

def test_load_pi05():
    # 配置
    model_path = "outputs/train/task1/pi05/run_20260203_194626/epochbest"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    print(f"Path exists: {Path(model_path).exists()}")
    
    # 导入 Pi05 Policy
    print("\n[1/3] Importing CustomPI05PolicyWrapper...")
    from kuavo_train.wrapper.policy.pi05.PI05PolicyWrapper import CustomPI05PolicyWrapper
    print("✓ Import successful")
    
    # 加载模型
    print("\n[2/3] Loading model (this may take a while for 9GB model)...")
    policy = CustomPI05PolicyWrapper.from_pretrained(Path(model_path), strict=True)
    print("✓ Model loaded")
    
    # 移动到设备
    print(f"\n[3/3] Moving to {device}...")
    policy = policy.to(device)
    policy.eval()
    print("✓ Model ready for inference")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✓ Pi05 model loaded successfully!")
    return policy

if __name__ == "__main__":
    test_load_pi05()
