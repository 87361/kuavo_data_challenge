#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线Action对比测试工具

目的：分析模型输出的action序列平滑度，帮助诊断动作不流畅的原因

测试内容：
1. 使用合成observation序列测试模型推理
2. 分析action序列的平滑度指标（jerk, 相邻帧差异等）
3. 测试temporal_ensemble的效果
4. 可选：对比不同配置下的action输出

使用方法:
    cd /home/yly/ICRA-kuavo/kuavo_data_challenge
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kdc_icra
    export PYTHONPATH="$PWD:$PWD/third_party/lerobot/src:$PYTHONPATH"
    
    python kuavo_deploy/utils/action_comparison_test.py \
        --model_path outputs/train/task1/act/run_20260202_223813/epochbest \
        --num_steps 100
"""

import argparse
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# 必须在导入lerobot之前应用patches以支持DEPTH feature type
import lerobot_patches.custom_patches  # noqa: F401

import numpy as np
import torch
from torch import Tensor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("action_comparison_test")


def create_observation_sequence(
    config,
    device: torch.device,
    num_steps: int,
    noise_level: float = 0.01
) -> List[Dict[str, Tensor]]:
    """创建一个模拟的observation序列
    
    Args:
        config: 模型配置
        device: 目标设备
        num_steps: 序列长度
        noise_level: 添加的噪声级别（模拟真实观测的变化）
        
    Returns:
        observation序列列表
    """
    obs_sequence = []
    
    # 创建基础observation
    base_obs = {}
    if hasattr(config, 'input_features') and config.input_features:
        for key, feature in config.input_features.items():
            if hasattr(feature, 'shape'):
                shape = feature.shape
            elif isinstance(feature, dict) and 'shape' in feature:
                shape = feature['shape']
            else:
                continue
            
            # 创建基础tensor
            base_obs[key] = torch.rand(
                (1,) + tuple(shape),
                dtype=torch.float32,
                device=device,
            )
    
    # 生成序列，添加小幅度变化模拟真实观测
    for i in range(num_steps):
        obs = {}
        for key, base_tensor in base_obs.items():
            # 添加小噪声，模拟观测的微小变化
            noise = torch.randn_like(base_tensor) * noise_level
            obs[key] = base_tensor + noise
        obs_sequence.append(obs)
    
    return obs_sequence


def compute_action_smoothness_metrics(actions: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
    """计算action序列的平滑度指标
    
    Args:
        actions: action序列 [num_steps, action_dim]
        dt: 时间步长（秒）
        
    Returns:
        平滑度指标字典
    """
    num_steps, action_dim = actions.shape
    
    metrics = {}
    
    # 1. 相邻帧差异 (velocity)
    if num_steps > 1:
        velocity = np.diff(actions, axis=0) / dt
        metrics['mean_velocity'] = np.mean(np.abs(velocity))
        metrics['max_velocity'] = np.max(np.abs(velocity))
        metrics['velocity_std'] = np.std(velocity)
    
    # 2. 加速度 (acceleration)
    if num_steps > 2:
        acceleration = np.diff(velocity, axis=0) / dt
        metrics['mean_acceleration'] = np.mean(np.abs(acceleration))
        metrics['max_acceleration'] = np.max(np.abs(acceleration))
        metrics['acceleration_std'] = np.std(acceleration)
    
    # 3. Jerk (d³x/dt³) - 衡量动作的突变程度
    if num_steps > 3:
        jerk = np.diff(acceleration, axis=0) / dt
        metrics['mean_jerk'] = np.mean(np.abs(jerk))
        metrics['max_jerk'] = np.max(np.abs(jerk))
        metrics['jerk_std'] = np.std(jerk)
    
    # 4. 总变化量
    metrics['total_variation'] = np.sum(np.abs(np.diff(actions, axis=0)))
    
    # 5. 每个维度的统计
    metrics['action_mean'] = np.mean(actions, axis=0).tolist()
    metrics['action_std'] = np.std(actions, axis=0).tolist()
    metrics['action_range'] = (np.max(actions, axis=0) - np.min(actions, axis=0)).tolist()
    
    return metrics


def run_inference_sequence(
    policy,
    preprocessor,
    postprocessor,
    obs_sequence: List[Dict[str, Tensor]],
    device: torch.device,
    reset_between_steps: bool = False
) -> Tuple[np.ndarray, List[float]]:
    """运行一个observation序列的推理
    
    Args:
        policy: 策略模型
        preprocessor: 预处理器
        postprocessor: 后处理器
        obs_sequence: observation序列
        device: 设备
        reset_between_steps: 是否每步重置policy状态
        
    Returns:
        (actions数组, 推理时间列表)
    """
    actions = []
    inference_times = []
    
    # 重置policy状态
    if hasattr(policy, 'reset'):
        policy.reset()
    
    for i, obs in enumerate(obs_sequence):
        if reset_between_steps and hasattr(policy, 'reset'):
            policy.reset()
        
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            # 预处理
            if preprocessor is not None:
                processed_obs = preprocessor(obs)
            else:
                processed_obs = obs
            
            # 推理
            try:
                action = policy.select_action(processed_obs)
            except Exception:
                action = policy.predict_action_chunk(processed_obs)
            
            # 后处理
            if postprocessor is not None:
                action = postprocessor(action)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = time.perf_counter() - start_time
        inference_times.append(inference_time)
        
        # 转换为numpy
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)
        
        # 确保是1D或2D
        if action_np.ndim == 3:
            action_np = action_np[0, 0]  # [batch, chunk, dim] -> [dim]
        elif action_np.ndim == 2:
            action_np = action_np[0]  # [batch, dim] -> [dim]
        
        actions.append(action_np)
    
    return np.array(actions), inference_times


def analyze_temporal_ensemble_effect(
    policy,
    preprocessor,
    obs_sequence: List[Dict[str, Tensor]],
    device: torch.device
) -> Dict[str, any]:
    """分析temporal_ensemble的效果
    
    Returns:
        分析结果字典
    """
    results = {}
    
    # 检查是否启用temporal_ensemble
    if hasattr(policy.config, 'temporal_ensemble_coeff'):
        coeff = policy.config.temporal_ensemble_coeff
        results['temporal_ensemble_enabled'] = coeff is not None
        results['temporal_ensemble_coeff'] = coeff
    else:
        results['temporal_ensemble_enabled'] = False
        results['temporal_ensemble_coeff'] = None
    
    # 检查action队列配置
    if hasattr(policy.config, 'n_action_steps'):
        results['n_action_steps'] = policy.config.n_action_steps
    if hasattr(policy.config, 'chunk_size'):
        results['chunk_size'] = policy.config.chunk_size
    
    return results


def print_analysis_report(
    smoothness_metrics: Dict[str, float],
    inference_times: List[float],
    temporal_config: Dict[str, any],
    actions: np.ndarray
):
    """打印分析报告"""
    
    print("\n" + "=" * 70)
    print("                    Action平滑度分析报告")
    print("=" * 70)
    
    # 配置信息
    print("\n【模型配置】")
    print("-" * 50)
    print(f"  temporal_ensemble_enabled: {temporal_config.get('temporal_ensemble_enabled', 'N/A')}")
    print(f"  temporal_ensemble_coeff: {temporal_config.get('temporal_ensemble_coeff', 'N/A')}")
    print(f"  chunk_size: {temporal_config.get('chunk_size', 'N/A')}")
    print(f"  n_action_steps: {temporal_config.get('n_action_steps', 'N/A')}")
    
    # 推理性能
    print("\n【推理性能】")
    print("-" * 50)
    print(f"  平均推理时间: {np.mean(inference_times)*1000:.2f} ms")
    print(f"  推理时间标准差: {np.std(inference_times)*1000:.2f} ms")
    print(f"  最大推理时间: {np.max(inference_times)*1000:.2f} ms")
    print(f"  最小推理时间: {np.min(inference_times)*1000:.2f} ms")
    
    # 平滑度指标
    print("\n【平滑度指标】 (值越小越平滑)")
    print("-" * 50)
    
    if 'mean_velocity' in smoothness_metrics:
        print(f"  平均速度 (|Δaction/Δt|): {smoothness_metrics['mean_velocity']:.6f}")
        print(f"  最大速度: {smoothness_metrics['max_velocity']:.6f}")
        print(f"  速度标准差: {smoothness_metrics['velocity_std']:.6f}")
    
    if 'mean_acceleration' in smoothness_metrics:
        print(f"\n  平均加速度: {smoothness_metrics['mean_acceleration']:.6f}")
        print(f"  最大加速度: {smoothness_metrics['max_acceleration']:.6f}")
    
    if 'mean_jerk' in smoothness_metrics:
        print(f"\n  平均Jerk (突变度): {smoothness_metrics['mean_jerk']:.6f}")
        print(f"  最大Jerk: {smoothness_metrics['max_jerk']:.6f}")
    
    print(f"\n  总变化量: {smoothness_metrics['total_variation']:.6f}")
    
    # Action统计
    print("\n【Action统计】")
    print("-" * 50)
    print(f"  序列长度: {actions.shape[0]}")
    print(f"  Action维度: {actions.shape[1]}")
    
    action_mean = smoothness_metrics.get('action_mean', [])
    action_std = smoothness_metrics.get('action_std', [])
    action_range = smoothness_metrics.get('action_range', [])
    
    if len(action_mean) <= 8:
        print(f"\n  各维度均值: {[f'{x:.4f}' for x in action_mean]}")
        print(f"  各维度标准差: {[f'{x:.4f}' for x in action_std]}")
        print(f"  各维度范围: {[f'{x:.4f}' for x in action_range]}")
    else:
        print(f"\n  均值范围: [{min(action_mean):.4f}, {max(action_mean):.4f}]")
        print(f"  标准差范围: [{min(action_std):.4f}, {max(action_std):.4f}]")
        print(f"  范围的范围: [{min(action_range):.4f}, {max(action_range):.4f}]")
    
    # 诊断建议
    print("\n【诊断分析】")
    print("-" * 50)
    
    # 分析平滑度
    if 'mean_jerk' in smoothness_metrics:
        jerk = smoothness_metrics['mean_jerk']
        if jerk > 1.0:
            print("  ⚠️  Jerk值较高，action序列存在明显突变")
            print("     可能原因: temporal_ensemble未生效或配置不当")
        elif jerk > 0.1:
            print("  ⚡ Jerk值中等，action序列有一定波动")
        else:
            print("  ✓  Jerk值较低，action序列较为平滑")
    
    # 分析temporal_ensemble
    if temporal_config.get('temporal_ensemble_enabled'):
        coeff = temporal_config.get('temporal_ensemble_coeff', 0)
        if coeff < 0:
            print(f"\n  ℹ️  temporal_ensemble_coeff={coeff} (负值)")
            print("     这意味着更重视新的预测，可能导致响应快但不够平滑")
        elif coeff > 0:
            print(f"\n  ℹ️  temporal_ensemble_coeff={coeff} (正值)")
            print("     这意味着更重视历史预测，动作会更平滑但响应较慢")
    else:
        print("\n  ℹ️  temporal_ensemble未启用，使用action queue缓存模式")
    
    # 对比推理时间和控制频率
    avg_inference_ms = np.mean(inference_times) * 1000
    target_freq_10hz = 100  # 10Hz = 100ms per step
    target_freq_20hz = 50   # 20Hz = 50ms per step
    
    print(f"\n  推理时间 vs 目标频率:")
    print(f"    - 平均推理: {avg_inference_ms:.1f}ms")
    print(f"    - 10Hz要求: <{target_freq_10hz}ms {'✓' if avg_inference_ms < target_freq_10hz else '✗'}")
    print(f"    - 20Hz要求: <{target_freq_20hz}ms {'✓' if avg_inference_ms < target_freq_20hz else '✗'}")
    
    print("\n" + "=" * 70)


def convert_to_serializable(obj):
    """将对象转换为JSON可序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def save_results(
    output_path: Path,
    actions: np.ndarray,
    smoothness_metrics: Dict,
    inference_times: List[float],
    temporal_config: Dict
):
    """保存测试结果"""
    results = {
        'smoothness_metrics': convert_to_serializable(smoothness_metrics),
        'inference_times': [float(t) for t in inference_times],
        'temporal_config': convert_to_serializable(temporal_config),
        'actions_shape': list(actions.shape),
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存actions为numpy文件
    np.save(output_path.with_suffix('.npy'), actions)
    
    log.info(f"结果已保存到: {output_path}")


def run_ablation_test(
    model_path: Path,
    pretrained_path: Path,
    device: torch.device,
    num_steps: int,
    noise_level: float,
    policy_type: str,
    output_dir: Path,
    warmup_steps: int
):
    """运行temporal_ensemble消融测试"""
    
    # 测试配置列表
    test_configs = [
        {"name": "original", "temporal_ensemble_coeff": None},  # 使用原始配置
        {"name": "disabled", "temporal_ensemble_coeff": "disabled"},  # 禁用
        {"name": "positive_0.01", "temporal_ensemble_coeff": 0.01},  # 正值
        {"name": "negative_0.1", "temporal_ensemble_coeff": -0.1},  # 负值
    ]
    
    results_summary = []
    
    for config in test_configs:
        config_name = config["name"]
        coeff = config["temporal_ensemble_coeff"]
        
        print(f"\n{'='*70}")
        print(f"测试配置: {config_name}")
        print(f"temporal_ensemble_coeff: {coeff}")
        print(f"{'='*70}")
        
        # 加载模型
        try:
            if policy_type == "act":
                from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
                policy = CustomACTPolicyWrapper.from_pretrained(model_path, strict=True)
            elif policy_type == "pi05":
                from kuavo_train.wrapper.policy.pi05.PI05PolicyWrapper import CustomPI05PolicyWrapper
                policy = CustomPI05PolicyWrapper.from_pretrained(model_path, strict=True)
            
            # 修改temporal_ensemble配置
            if coeff == "disabled":
                policy.config.temporal_ensemble_coeff = None
            elif coeff is not None:
                policy.config.temporal_ensemble_coeff = coeff
            
            # 重新初始化temporal_ensembler（如果需要）
            if hasattr(policy, 'temporal_ensembler') and policy.config.temporal_ensemble_coeff is not None:
                from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
                policy.temporal_ensembler = ACTTemporalEnsembler(
                    policy.config.temporal_ensemble_coeff,
                    policy.config.chunk_size
                )
            
            policy.eval()
            policy.to(device)
            
        except Exception as e:
            log.error(f"配置 {config_name} 加载失败: {e}")
            continue
        
        # 加载processor
        try:
            from lerobot.policies.factory import make_pre_post_processors
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=policy.config,
                pretrained_path=str(pretrained_path)
            )
        except Exception as e:
            log.warning(f"加载processor失败: {e}")
            preprocessor, postprocessor = None, None
        
        # 创建observation序列
        obs_sequence = create_observation_sequence(policy.config, device, num_steps, noise_level)
        
        # 预热
        warmup_obs = create_observation_sequence(policy.config, device, warmup_steps, noise_level)
        _, _ = run_inference_sequence(policy, preprocessor, postprocessor, warmup_obs, device)
        
        # 重置
        if hasattr(policy, 'reset'):
            policy.reset()
        
        # 运行测试
        actions, inference_times = run_inference_sequence(
            policy, preprocessor, postprocessor, obs_sequence, device
        )
        
        # 计算指标
        smoothness_metrics = compute_action_smoothness_metrics(actions, dt=0.1)
        temporal_config = analyze_temporal_ensemble_effect(policy, preprocessor, obs_sequence, device)
        
        # 记录结果
        result = {
            "config_name": config_name,
            "temporal_ensemble_coeff": coeff,
            "mean_jerk": smoothness_metrics.get('mean_jerk', 0),
            "max_jerk": smoothness_metrics.get('max_jerk', 0),
            "mean_velocity": smoothness_metrics.get('mean_velocity', 0),
            "mean_inference_ms": np.mean(inference_times) * 1000,
            "total_variation": smoothness_metrics.get('total_variation', 0),
        }
        results_summary.append(result)
        
        # 打印简要结果
        print(f"  平均Jerk: {result['mean_jerk']:.6f}")
        print(f"  最大Jerk: {result['max_jerk']:.6f}")
        print(f"  平均推理时间: {result['mean_inference_ms']:.2f} ms")
        
        # 释放内存
        del policy
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # 打印对比总结
    print("\n" + "=" * 70)
    print("                   消融测试结果对比")
    print("=" * 70)
    print(f"{'配置':<20} {'Jerk(平均)':<15} {'Jerk(最大)':<15} {'推理(ms)':<12} {'平滑度'}")
    print("-" * 70)
    
    for r in results_summary:
        smoothness = "优秀" if r['mean_jerk'] < 0.1 else ("良好" if r['mean_jerk'] < 0.5 else "一般")
        print(f"{r['config_name']:<20} {r['mean_jerk']:<15.6f} {r['max_jerk']:<15.6f} {r['mean_inference_ms']:<12.2f} {smoothness}")
    
    print("=" * 70)
    
    # 保存对比结果
    summary_path = output_dir / "ablation_test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(convert_to_serializable(results_summary), f, indent=2)
    log.info(f"消融测试结果已保存到: {summary_path}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description="离线Action对比测试工具")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/train/task1/act/run_20260202_223813/epochbest",
        help="模型路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="测试序列长度"
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.01,
        help="observation噪声级别"
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="act",
        choices=["act", "pi05"],
        help="策略类型"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/analysis",
        help="输出目录"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="预热步数"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="运行temporal_ensemble消融测试"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    log.info(f"使用设备: {device}")
    
    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        log.error(f"模型路径不存在: {model_path}")
        sys.exit(1)
    
    # 获取pretrained_path
    if model_path.name.startswith("epoch"):
        pretrained_path = model_path.parent
    else:
        pretrained_path = model_path
    
    log.info(f"模型路径: {model_path}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 如果是消融测试模式
    if args.ablation:
        log.info("运行temporal_ensemble消融测试...")
        run_ablation_test(
            model_path, pretrained_path, device, 
            args.num_steps, args.noise_level, args.policy_type,
            output_dir, args.warmup_steps
        )
        return
    
    # 加载策略模型
    log.info("加载策略模型...")
    try:
        if args.policy_type == "act":
            from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
            policy = CustomACTPolicyWrapper.from_pretrained(model_path, strict=True)
        elif args.policy_type == "pi05":
            from kuavo_train.wrapper.policy.pi05.PI05PolicyWrapper import CustomPI05PolicyWrapper
            policy = CustomPI05PolicyWrapper.from_pretrained(model_path, strict=True)
        
        policy.eval()
        policy.to(device)
        log.info(f"策略模型加载成功: {type(policy).__name__}")
    except Exception as e:
        log.error(f"加载策略模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 加载processor
    log.info("加载 preprocessor/postprocessor...")
    try:
        from lerobot.policies.factory import make_pre_post_processors
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(pretrained_path)
        )
        log.info("Processor加载成功")
    except Exception as e:
        log.warning(f"加载processor失败: {e}")
        preprocessor, postprocessor = None, None
    
    # 创建observation序列
    log.info(f"创建observation序列 (长度={args.num_steps}, 噪声={args.noise_level})...")
    obs_sequence = create_observation_sequence(
        policy.config, device, args.num_steps, args.noise_level
    )
    
    # 分析temporal_ensemble配置
    temporal_config = analyze_temporal_ensemble_effect(policy, preprocessor, obs_sequence, device)
    
    # 预热
    log.info(f"预热 ({args.warmup_steps} steps)...")
    warmup_obs = create_observation_sequence(policy.config, device, args.warmup_steps, args.noise_level)
    _, _ = run_inference_sequence(policy, preprocessor, postprocessor, warmup_obs, device)
    
    # 重置policy状态
    if hasattr(policy, 'reset'):
        policy.reset()
    
    # 运行推理序列
    log.info(f"运行推理序列 ({args.num_steps} steps)...")
    actions, inference_times = run_inference_sequence(
        policy, preprocessor, postprocessor, obs_sequence, device
    )
    
    log.info(f"Actions shape: {actions.shape}")
    
    # 计算平滑度指标
    log.info("计算平滑度指标...")
    smoothness_metrics = compute_action_smoothness_metrics(actions, dt=0.1)  # 假设10Hz
    
    # 打印报告
    print_analysis_report(smoothness_metrics, inference_times, temporal_config, actions)
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"action_analysis_{args.policy_type}.json"
    save_results(output_path, actions, smoothness_metrics, inference_times, temporal_config)
    
    log.info("测试完成!")


if __name__ == "__main__":
    main()
