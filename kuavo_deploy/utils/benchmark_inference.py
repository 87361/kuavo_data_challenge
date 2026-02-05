#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立的性能验证Benchmark脚本

验证文档中关于 main 分支 vs icra 分支性能差异的猜测：
1. preprocessor/postprocessor 额外开销
2. ObsBuffer 中间层延迟 (跳过，需要ROS环境)
3. 归一化位置差异的影响

使用方法:
    cd /home/yly/ICRA-kuavo/kuavo_data_challenge
    python kuavo_deploy/utils/benchmark_inference.py \
        --model_path outputs/train/task1/act/run_20260202_223813/epochbest

注意: 此脚本不修改任何现有代码，测试完成后可直接删除。
"""

import argparse
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import statistics

# 必须在导入lerobot之前应用patches以支持DEPTH feature type
import lerobot_patches.custom_patches  # noqa: F401

import torch
from torch import Tensor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("benchmark_inference")


def create_dummy_observation_from_config(config, device: torch.device) -> Dict[str, Tensor]:
    """根据模型配置创建dummy observation
    
    Args:
        config: 模型配置，包含input_features信息
        device: 目标设备
        
    Returns:
        dummy observation字典
    """
    dummy_obs = {}
    
    if hasattr(config, 'input_features') and config.input_features:
        for key, feature in config.input_features.items():
            if hasattr(feature, 'shape'):
                shape = feature.shape
            elif isinstance(feature, dict) and 'shape' in feature:
                shape = feature['shape']
            else:
                log.warning(f"无法获取 {key} 的shape，跳过")
                continue
                
            feature_type = None
            if hasattr(feature, 'type'):
                feature_type = feature.type
            elif isinstance(feature, dict) and 'type' in feature:
                feature_type = feature['type']
            
            # 根据feature类型创建tensor
            if feature_type in ['VISUAL', 'RGB']:
                # 图像: [B, C, H, W]
                dummy_obs[key] = torch.rand(
                    (1,) + tuple(shape),
                    dtype=torch.float32,
                    device=device,
                )
            elif feature_type == 'DEPTH':
                # 深度: [B, C, H, W]
                dummy_obs[key] = torch.rand(
                    (1,) + tuple(shape),
                    dtype=torch.float32,
                    device=device,
                )
            elif feature_type == 'STATE':
                # 状态: [B, state_dim]
                dummy_obs[key] = torch.rand(
                    (1,) + tuple(shape),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                # 默认处理
                dummy_obs[key] = torch.rand(
                    (1,) + tuple(shape),
                    dtype=torch.float32,
                    device=device,
                )
                
    return dummy_obs


def benchmark_processor(
    processor,
    input_data: Dict[str, Any],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: torch.device = None
) -> Dict[str, float]:
    """测量processor的执行时间
    
    Args:
        processor: preprocessor或postprocessor
        input_data: 输入数据
        num_runs: 计时运行次数
        warmup_runs: 预热次数
        device: GPU设备(用于同步)
        
    Returns:
        时间统计字典
    """
    # 预热
    for _ in range(warmup_runs):
        _ = processor(input_data)
    
    # 同步GPU
    if device and device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = processor(input_data)
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        "mean_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
    }


def benchmark_inference(
    policy,
    dummy_obs: Dict[str, Tensor],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: torch.device = None
) -> Dict[str, float]:
    """测量纯模型推理时间（不含processor）
    
    Args:
        policy: 策略模型
        dummy_obs: dummy observation（已经过preprocessor处理，或原始）
        num_runs: 计时运行次数
        warmup_runs: 预热次数
        device: GPU设备
        
    Returns:
        时间统计字典
    """
    # 预热
    for _ in range(warmup_runs):
        with torch.inference_mode():
            try:
                _ = policy.select_action(dummy_obs)
            except Exception:
                try:
                    _ = policy.predict_action_chunk(dummy_obs)
                except Exception:
                    pass
    
    # 同步GPU
    if device and device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    times = []
    for _ in range(num_runs):
        # 重置policy的action queue（如果有的话）
        if hasattr(policy, 'reset'):
            policy.reset()
            
        start = time.perf_counter()
        with torch.inference_mode():
            try:
                _ = policy.select_action(dummy_obs)
            except Exception:
                _ = policy.predict_action_chunk(dummy_obs)
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        "mean_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
    }


def benchmark_full_pipeline(
    policy,
    preprocessor,
    postprocessor,
    raw_obs: Dict[str, Tensor],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: torch.device = None
) -> Dict[str, float]:
    """测量完整推理管线时间（preprocessor + model + postprocessor）
    
    Args:
        policy: 策略模型
        preprocessor: 预处理器
        postprocessor: 后处理器
        raw_obs: 原始observation
        num_runs: 计时运行次数
        warmup_runs: 预热次数
        device: GPU设备
        
    Returns:
        时间统计字典
    """
    # 预热
    for _ in range(warmup_runs):
        with torch.inference_mode():
            processed_obs = preprocessor(raw_obs)
            try:
                action = policy.select_action(processed_obs)
            except Exception:
                action = policy.predict_action_chunk(processed_obs)
            _ = postprocessor(action)
    
    # 同步GPU
    if device and device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    times = []
    for _ in range(num_runs):
        if hasattr(policy, 'reset'):
            policy.reset()
            
        start = time.perf_counter()
        with torch.inference_mode():
            processed_obs = preprocessor(raw_obs)
            try:
                action = policy.select_action(processed_obs)
            except Exception:
                action = policy.predict_action_chunk(processed_obs)
            _ = postprocessor(action)
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return {
        "mean_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
    }


def print_results(
    preprocessor_stats: Optional[Dict[str, float]],
    postprocessor_stats: Optional[Dict[str, float]],
    inference_stats: Optional[Dict[str, float]],
    full_pipeline_stats: Optional[Dict[str, float]],
):
    """打印测试结果报告"""
    
    print("\n" + "=" * 60)
    print("           性能验证测试结果")
    print("=" * 60)
    
    # 测试1: Processor开销
    print("\n测试1: Preprocessor/Postprocessor 开销")
    print("-" * 40)
    
    if preprocessor_stats:
        print(f"  Preprocessor 平均耗时: {preprocessor_stats['mean_ms']:.3f} ms")
        print(f"    (标准差: {preprocessor_stats['std_ms']:.3f} ms, "
              f"中位数: {preprocessor_stats['median_ms']:.3f} ms)")
    else:
        print("  Preprocessor: [跳过]")
        
    if postprocessor_stats:
        print(f"  Postprocessor 平均耗时: {postprocessor_stats['mean_ms']:.3f} ms")
        print(f"    (标准差: {postprocessor_stats['std_ms']:.3f} ms, "
              f"中位数: {postprocessor_stats['median_ms']:.3f} ms)")
    else:
        print("  Postprocessor: [跳过]")
    
    if preprocessor_stats and postprocessor_stats:
        total_processor = preprocessor_stats['mean_ms'] + postprocessor_stats['mean_ms']
        print(f"  Processor 总开销: {total_processor:.3f} ms")
    
    # 测试2: ObsBuffer延迟
    print("\n测试2: ObsBuffer 延迟")
    print("-" * 40)
    print("  [跳过] 需要ROS环境，无法在独立benchmark中测试")
    
    # 测试3: 纯模型推理
    print("\n测试3: 纯模型推理时间")
    print("-" * 40)
    
    if inference_stats:
        print(f"  select_action 平均耗时: {inference_stats['mean_ms']:.3f} ms")
        print(f"    (标准差: {inference_stats['std_ms']:.3f} ms, "
              f"中位数: {inference_stats['median_ms']:.3f} ms)")
    else:
        print("  [跳过]")
    
    # 完整管线
    print("\n测试4: 完整推理管线")
    print("-" * 40)
    
    if full_pipeline_stats:
        print(f"  完整管线 平均耗时: {full_pipeline_stats['mean_ms']:.3f} ms")
        print(f"    (标准差: {full_pipeline_stats['std_ms']:.3f} ms, "
              f"中位数: {full_pipeline_stats['median_ms']:.3f} ms)")
    else:
        print("  [跳过]")
    
    # 结论
    print("\n" + "=" * 60)
    print("                   结论")
    print("=" * 60)
    
    if preprocessor_stats and postprocessor_stats and inference_stats:
        total_processor = preprocessor_stats['mean_ms'] + postprocessor_stats['mean_ms']
        total_with_processor = total_processor + inference_stats['mean_ms']
        processor_ratio = (total_processor / total_with_processor) * 100
        
        print(f"\n  Processor 开销占比: {processor_ratio:.1f}%")
        print(f"    - Preprocessor: {preprocessor_stats['mean_ms']:.3f} ms "
              f"({preprocessor_stats['mean_ms']/total_with_processor*100:.1f}%)")
        print(f"    - Postprocessor: {postprocessor_stats['mean_ms']:.3f} ms "
              f"({postprocessor_stats['mean_ms']/total_with_processor*100:.1f}%)")
        print(f"    - 纯推理: {inference_stats['mean_ms']:.3f} ms "
              f"({inference_stats['mean_ms']/total_with_processor*100:.1f}%)")
        
        # 猜测验证
        if processor_ratio > 10:
            print(f"\n  猜测验证: [成立] Processor开销占比 {processor_ratio:.1f}% > 10%，是显著开销")
        else:
            print(f"\n  猜测验证: [不成立] Processor开销占比 {processor_ratio:.1f}% <= 10%，开销较小")
    
    if full_pipeline_stats and inference_stats:
        overhead = full_pipeline_stats['mean_ms'] - inference_stats['mean_ms']
        print(f"\n  完整管线 vs 纯推理 额外开销: {overhead:.3f} ms")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="性能验证Benchmark测试")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/train/task1/act/run_20260202_223813/epochbest",
        help="模型路径 (epochbest目录)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="计时运行次数"
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=10,
        help="预热运行次数"
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        default="act",
        choices=["act", "pi05"],
        help="策略类型"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    log.info(f"使用设备: {device}")
    
    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        log.error(f"模型路径不存在: {model_path}")
        sys.exit(1)
    
    # 获取pretrained_path (epochbest的父目录，包含processor配置)
    if model_path.name.startswith("epoch"):
        pretrained_path = model_path.parent
    else:
        pretrained_path = model_path
    
    log.info(f"模型路径: {model_path}")
    log.info(f"Processor配置路径: {pretrained_path}")
    
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
        log.error(f"加载processor失败: {e}")
        import traceback
        traceback.print_exc()
        preprocessor, postprocessor = None, None
    
    # 创建dummy observation
    log.info("创建dummy observation...")
    raw_obs = create_dummy_observation_from_config(policy.config, device)
    log.info(f"Dummy observation keys: {list(raw_obs.keys())}")
    for k, v in raw_obs.items():
        log.info(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
    
    # 运行测试
    log.info(f"\n开始性能测试 (warmup={args.warmup_runs}, runs={args.num_runs})...")
    
    preprocessor_stats = None
    postprocessor_stats = None
    inference_stats = None
    full_pipeline_stats = None
    
    # 测试1: Preprocessor开销
    if preprocessor is not None:
        log.info("测试1a: 测量 Preprocessor 开销...")
        try:
            preprocessor_stats = benchmark_processor(
                preprocessor, raw_obs, args.num_runs, args.warmup_runs, device
            )
            log.info(f"  Preprocessor: {preprocessor_stats['mean_ms']:.3f} ms")
        except Exception as e:
            log.warning(f"Preprocessor测试失败: {e}")
    
    # 测试1b: Postprocessor开销
    if postprocessor is not None:
        log.info("测试1b: 测量 Postprocessor 开销...")
        try:
            # 创建一个dummy action用于测试postprocessor
            # PolicyAction 是 torch.Tensor 的别名 (见 lerobot/processor/core.py)
            action_shape = policy.config.output_features.get("action", {})
            if hasattr(action_shape, 'shape'):
                action_dim = action_shape.shape[0]
            elif isinstance(action_shape, dict) and 'shape' in action_shape:
                action_dim = action_shape['shape'][0]
            else:
                action_dim = 16  # 默认
            
            # 获取chunk_size (ACT模型通常输出action chunks)
            chunk_size = getattr(policy.config, 'chunk_size', 100)
            
            # PolicyAction 格式: (batch, chunk_size, action_dim) 或 (batch, action_dim)
            dummy_action = torch.rand(
                (1, chunk_size, action_dim), dtype=torch.float32, device=device
            )
            
            postprocessor_stats = benchmark_processor(
                postprocessor, dummy_action, args.num_runs, args.warmup_runs, device
            )
            log.info(f"  Postprocessor: {postprocessor_stats['mean_ms']:.3f} ms")
        except Exception as e:
            log.warning(f"Postprocessor测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 测试3: 纯模型推理
    log.info("测试3: 测量纯模型推理时间...")
    try:
        # 如果有preprocessor，先处理observation
        if preprocessor is not None:
            processed_obs = preprocessor(raw_obs)
        else:
            processed_obs = raw_obs
        
        inference_stats = benchmark_inference(
            policy, processed_obs, args.num_runs, args.warmup_runs, device
        )
        log.info(f"  纯推理: {inference_stats['mean_ms']:.3f} ms")
    except Exception as e:
        log.warning(f"推理测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试4: 完整管线
    if preprocessor is not None and postprocessor is not None:
        log.info("测试4: 测量完整推理管线时间...")
        try:
            full_pipeline_stats = benchmark_full_pipeline(
                policy, preprocessor, postprocessor,
                raw_obs, args.num_runs, args.warmup_runs, device
            )
            log.info(f"  完整管线: {full_pipeline_stats['mean_ms']:.3f} ms")
        except Exception as e:
            log.warning(f"完整管线测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印结果
    print_results(
        preprocessor_stats,
        postprocessor_stats,
        inference_stats,
        full_pipeline_stats
    )
    
    log.info("测试完成!")


if __name__ == "__main__":
    main()
