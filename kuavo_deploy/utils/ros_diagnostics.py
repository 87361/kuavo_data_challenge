#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROS环境深入诊断脚本

在ROS仿真环境中诊断动作不流畅的真实原因，通过测量各环节延迟和录制action序列分析。

诊断内容：
1. ObsBuffer各环节延迟
2. 端到端推理循环延迟
3. Action序列录制与平滑度分析

使用方法:
    # 步骤1: 启动仿真器 (在 kuavo-ros-opensource 仓库中)
    
    # 步骤2: 运行诊断脚本
    cd /home/yly/ICRA-kuavo/kuavo_data_challenge
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate kdc_icra
    export PYTHONPATH="$PWD:$PWD/third_party/lerobot/src:$PYTHONPATH"
    
    python kuavo_deploy/utils/ros_diagnostics.py \
        --model_path outputs/train/task1/act/run_20260202_223813/epochbest \
        --config configs/deploy/kuavo_env.yaml \
        --num_steps 100 \
        --save_actions
"""

import argparse
import time
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import json
import statistics

# 必须在导入lerobot之前应用patches
import lerobot_patches.custom_patches  # noqa: F401

import numpy as np
import torch
from torch import Tensor

# ROS imports - will fail if ROS not available
try:
    import rospy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("警告: ROS不可用，部分功能将被禁用")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("ros_diagnostics")


class TimingStats:
    """收集和统计时间数据"""
    
    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []
    
    def add(self, time_ms: float):
        self.times.append(time_ms)
    
    def get_stats(self) -> Dict[str, float]:
        if not self.times:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
        return {
            "mean": statistics.mean(self.times),
            "std": statistics.stdev(self.times) if len(self.times) > 1 else 0,
            "min": min(self.times),
            "max": max(self.times),
            "count": len(self.times),
        }


class DiagnosticObsBuffer:
    """带诊断功能的ObsBuffer包装器"""
    
    def __init__(self, obs_buffer):
        self.obs_buffer = obs_buffer
        self.get_obs_times = TimingStats("get_obs")
        self.get_aligned_obs_times = TimingStats("get_aligned_obs")
        self.get_latest_obs_times = TimingStats("get_latest_obs")
    
    def get_obs_timed(self, frame_alignment: bool, ros_rate: float, ratio: float = 1.0):
        """带计时的观测获取"""
        start = time.perf_counter()
        
        if frame_alignment:
            inner_start = time.perf_counter()
            obs = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=1/ros_rate, ratio=ratio)
            self.get_aligned_obs_times.add((time.perf_counter() - inner_start) * 1000)
            
            if obs is None or not all(v is not None for v in obs.values()):
                inner_start = time.perf_counter()
                obs = self.obs_buffer.get_aligned_obs(reference_keys=None, max_dt=float('inf'), ratio=ratio)
                self.get_aligned_obs_times.add((time.perf_counter() - inner_start) * 1000)
        else:
            inner_start = time.perf_counter()
            obs = self.obs_buffer.get_latest_obs()
            self.get_latest_obs_times.add((time.perf_counter() - inner_start) * 1000)
        
        total_time = (time.perf_counter() - start) * 1000
        self.get_obs_times.add(total_time)
        
        return obs, total_time


def compute_action_smoothness(actions: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
    """计算action序列的平滑度指标"""
    num_steps, action_dim = actions.shape
    metrics = {}
    
    if num_steps > 1:
        velocity = np.diff(actions, axis=0) / dt
        metrics['mean_velocity'] = float(np.mean(np.abs(velocity)))
        metrics['max_velocity'] = float(np.max(np.abs(velocity)))
    
    if num_steps > 2:
        acceleration = np.diff(velocity, axis=0) / dt
        metrics['mean_acceleration'] = float(np.mean(np.abs(acceleration)))
        metrics['max_acceleration'] = float(np.max(np.abs(acceleration)))
    
    if num_steps > 3:
        jerk = np.diff(acceleration, axis=0) / dt
        metrics['mean_jerk'] = float(np.mean(np.abs(jerk)))
        metrics['max_jerk'] = float(np.max(np.abs(jerk)))
    
    metrics['total_variation'] = float(np.sum(np.abs(np.diff(actions, axis=0))))
    
    return metrics


def run_e2e_diagnosis(
    env,
    policy,
    preprocessor,
    postprocessor,
    diag_obs_buffer: DiagnosticObsBuffer,
    num_steps: int,
    device: torch.device,
    frame_alignment: bool,
    ros_rate: float,
    ratio: float = 1.0,
) -> Tuple[Dict[str, TimingStats], np.ndarray]:
    """运行端到端诊断
    
    Returns:
        (timing_stats字典, action序列)
    """
    # 初始化计时统计
    timing_stats = {
        "obs_get": TimingStats("观测获取"),
        "preprocess": TimingStats("预处理"),
        "inference": TimingStats("模型推理"),
        "postprocess": TimingStats("后处理"),
        "action_exec": TimingStats("动作执行"),
        "step_total": TimingStats("每步总耗时"),
    }
    
    actions = []
    
    # 重置policy
    if hasattr(policy, 'reset'):
        policy.reset()
    
    log.info(f"开始端到端诊断 ({num_steps} steps)...")
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        # 1. 获取观测
        obs_start = time.perf_counter()
        raw_obs, obs_time = diag_obs_buffer.get_obs_timed(frame_alignment, ros_rate, ratio)
        timing_stats["obs_get"].add(obs_time)
        
        # 转换观测格式（与 KuavoBaseRosEnv.get_obs 保持一致）
        obs = {}
        arm_state = {}
        for k, v in raw_obs.items():
            if 'depth' in k:
                obs[f"observation.{k}"] = v
            elif 'cam' in k:
                obs[f"observation.images.{k}"] = v
            else:
                arm_state[k] = v
        
        # 构建 observation.state（与 KuavoBaseRosEnv.get_obs 逻辑一致）
        # arm_state_keys 默认为 ["joint_q", "gripper"]
        # which_arm 默认为 "both"
        arm_data = {"left": [], "right": []}
        
        # 处理 joint_q：从 28 维完整关节中提取双臂 14 个关节（索引 12-26）
        if 'joint_q' in arm_state:
            joint_q = arm_state['joint_q']
            if isinstance(joint_q, (list, tuple)):
                joint_q = list(joint_q)
            elif isinstance(joint_q, np.ndarray):
                joint_q = joint_q.tolist()
            
            # 提取双臂关节（索引 12-26，共 14 个）
            if len(joint_q) >= 26:
                arm_joints = joint_q[12:26]  # 14 个关节
                mid = len(arm_joints) // 2   # 7
                arm_data["left"].append(arm_joints[:mid])   # 左臂 7 个关节
                arm_data["right"].append(arm_joints[mid:])  # 右臂 7 个关节
        
        # 处理 gripper：从末端执行器数据获取
        gripper_keys = ['rq2f85', 'leju_claw', 'qiangnao']
        gripper_data = None
        for gk in gripper_keys:
            if gk in arm_state:
                gd = arm_state[gk]
                if isinstance(gd, (list, tuple, np.ndarray)) and len(gd) > 0:
                    gripper_data = list(gd) if not isinstance(gd, np.ndarray) else gd.tolist()
                    break
        
        # 如果没有 gripper 数据，使用默认值（全关闭状态）
        if gripper_data is None:
            gripper_data = [0.0, 0.0]  # 默认夹爪关闭
        
        # 确保 gripper 只有 2 维
        if len(gripper_data) >= 2:
            gripper_data = gripper_data[:2]
        elif len(gripper_data) == 1:
            gripper_data = gripper_data * 2
        
        mid_g = len(gripper_data) // 2
        arm_data["left"].append(gripper_data[:mid_g] if mid_g > 0 else [gripper_data[0]])
        arm_data["right"].append(gripper_data[mid_g:] if mid_g > 0 else [gripper_data[-1]])
        
        # 拼接状态：左臂数据 + 右臂数据
        state_list = []
        for part in arm_data["left"]:
            state_list.extend(part)
        for part in arm_data["right"]:
            state_list.extend(part)
        
        if state_list:
            obs["observation.state"] = torch.tensor(
                state_list, dtype=torch.float32, device=device
            ).unsqueeze(0)
        
        # 2. 预处理
        preprocess_start = time.perf_counter()
        if preprocessor is not None:
            processed_obs = preprocessor(obs)
        else:
            processed_obs = obs
        timing_stats["preprocess"].add((time.perf_counter() - preprocess_start) * 1000)
        
        # 3. 模型推理
        inference_start = time.perf_counter()
        with torch.inference_mode():
            try:
                action = policy.select_action(processed_obs)
            except Exception as e:
                log.warning(f"select_action失败: {e}, 尝试predict_action_chunk")
                action = policy.predict_action_chunk(processed_obs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timing_stats["inference"].add((time.perf_counter() - inference_start) * 1000)
        
        # 4. 后处理
        postprocess_start = time.perf_counter()
        if postprocessor is not None:
            action = postprocessor(action)
        timing_stats["postprocess"].add((time.perf_counter() - postprocess_start) * 1000)
        
        # 转换action为numpy
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action)
        
        if action_np.ndim == 3:
            action_np = action_np[0, 0]
        elif action_np.ndim == 2:
            action_np = action_np[0]
        
        actions.append(action_np.copy())
        
        # 5. 动作执行 (模拟)
        exec_start = time.perf_counter()
        # 在真实环境中这里会调用 env.step(action)
        # 这里只是模拟延迟
        time.sleep(0.001)  # 最小延迟
        timing_stats["action_exec"].add((time.perf_counter() - exec_start) * 1000)
        
        # 总耗时
        step_total = (time.perf_counter() - step_start) * 1000
        timing_stats["step_total"].add(step_total)
        
        # 控制频率
        target_period = 1.0 / ros_rate
        elapsed = time.perf_counter() - step_start
        if elapsed < target_period:
            time.sleep(target_period - elapsed)
        
        if (step + 1) % 20 == 0:
            log.info(f"  Step {step+1}/{num_steps}, 本步耗时: {step_total:.1f}ms")
    
    return timing_stats, np.array(actions)


def print_diagnosis_report(
    obs_buffer_stats: DiagnosticObsBuffer,
    e2e_stats: Dict[str, TimingStats],
    action_smoothness: Dict[str, float],
    actions: np.ndarray,
    ros_rate: float,
):
    """打印诊断报告"""
    
    print("\n" + "=" * 70)
    print("                    ROS环境深入诊断报告")
    print("=" * 70)
    
    # ObsBuffer延迟分析
    print("\n【ObsBuffer延迟分析】")
    print("-" * 50)
    
    get_obs = obs_buffer_stats.get_obs_times.get_stats()
    get_latest = obs_buffer_stats.get_latest_obs_times.get_stats()
    get_aligned = obs_buffer_stats.get_aligned_obs_times.get_stats()
    
    print(f"  get_obs() 平均耗时: {get_obs['mean']:.2f} ms (std: {get_obs['std']:.2f} ms)")
    
    if get_latest['count'] > 0:
        print(f"  get_latest_obs() 平均耗时: {get_latest['mean']:.2f} ms")
    if get_aligned['count'] > 0:
        print(f"  get_aligned_obs() 平均耗时: {get_aligned['mean']:.2f} ms")
    
    # ObsBuffer内部计时详情（如果启用）
    if hasattr(obs_buffer_stats.obs_buffer, 'timing') and obs_buffer_stats.obs_buffer.timing.enabled:
        print("\n  [ObsBuffer内部计时详情]")
        internal_stats = obs_buffer_stats.obs_buffer.timing.get_all_stats()
        
        # 按类别分组显示
        callback_keys = [k for k in internal_stats.keys() if 'callback' in k]
        obs_get_keys = [k for k in internal_stats.keys() if 'aligned' in k or 'latest' in k]
        other_keys = [k for k in internal_stats.keys() if k not in callback_keys and k not in obs_get_keys]
        
        if callback_keys:
            print("    回调函数:")
            for key in sorted(callback_keys):
                s = internal_stats[key]
                print(f"      {key}: {s['mean']:.2f} ms (std: {s['std']:.2f}, n={s['count']})")
        
        if obs_get_keys:
            print("    观测获取:")
            for key in sorted(obs_get_keys):
                s = internal_stats[key]
                print(f"      {key}: {s['mean']:.2f} ms (std: {s['std']:.2f}, n={s['count']})")
        
        if other_keys:
            print("    其他:")
            for key in sorted(other_keys):
                s = internal_stats[key]
                print(f"      {key}: {s['mean']:.2f} ms (std: {s['std']:.2f}, n={s['count']})")
    
    # 端到端延迟分析
    print("\n【端到端推理延迟分析】")
    print("-" * 50)
    
    total_stats = e2e_stats["step_total"].get_stats()
    total_mean = total_stats['mean']
    
    components = ["obs_get", "preprocess", "inference", "postprocess", "action_exec"]
    component_names = ["观测获取", "预处理", "模型推理", "后处理", "动作执行"]
    
    for comp, name in zip(components, component_names):
        stats = e2e_stats[comp].get_stats()
        pct = (stats['mean'] / total_mean * 100) if total_mean > 0 else 0
        print(f"  {name}: {stats['mean']:.2f} ms ({pct:.1f}%)")
    
    print("  " + "-" * 40)
    print(f"  每步总耗时: {total_mean:.2f} ms (std: {total_stats['std']:.2f} ms)")
    
    actual_freq = 1000 / total_mean if total_mean > 0 else 0
    print(f"  实际控制频率: {actual_freq:.1f} Hz (目标: {ros_rate} Hz)")
    
    if actual_freq < ros_rate * 0.9:
        print(f"  ⚠️  实际频率低于目标的90%!")
    
    # Action序列分析
    print("\n【Action序列分析】")
    print("-" * 50)
    print(f"  录制步数: {actions.shape[0]}")
    print(f"  Action维度: {actions.shape[1]}")
    
    if 'mean_jerk' in action_smoothness:
        print(f"  平均Jerk: {action_smoothness['mean_jerk']:.6f}")
        print(f"  最大Jerk: {action_smoothness['max_jerk']:.6f}")
        print(f"  (对比离线测试基准: Jerk≈0.32)")
        
        # 判断平滑度
        jerk = action_smoothness['mean_jerk']
        if jerk < 0.5:
            smoothness = "良好"
        elif jerk < 1.0:
            smoothness = "一般"
        else:
            smoothness = "较差"
        print(f"  平滑度评级: {smoothness}")
    
    if 'mean_velocity' in action_smoothness:
        print(f"  平均速度: {action_smoothness['mean_velocity']:.6f}")
    
    print(f"  总变化量: {action_smoothness.get('total_variation', 0):.6f}")
    
    # 诊断结论
    print("\n【诊断结论】")
    print("-" * 50)
    
    # 找出瓶颈
    bottleneck = None
    max_pct = 0
    for comp, name in zip(components, component_names):
        stats = e2e_stats[comp].get_stats()
        pct = (stats['mean'] / total_mean * 100) if total_mean > 0 else 0
        if pct > max_pct:
            max_pct = pct
            bottleneck = name
    
    print(f"  主要瓶颈环节: {bottleneck} ({max_pct:.1f}%)")
    
    # 建议
    inference_stats = e2e_stats["inference"].get_stats()
    obs_stats = e2e_stats["obs_get"].get_stats()
    
    if inference_stats['mean'] > 40:
        print("  建议: 模型推理耗时较高，考虑使用torch.compile或模型量化")
    if obs_stats['mean'] > 10:
        print("  建议: 观测获取耗时较高，考虑降低图像分辨率或使用GPU解码")
    if actual_freq < ros_rate * 0.9:
        print("  建议: 实际频率未达标，考虑使用异步推理(VLASH)")
    
    if 'mean_jerk' in action_smoothness:
        jerk = action_smoothness['mean_jerk']
        if jerk > 1.0:
            print("  建议: Jerk值较高，检查模型训练数据质量或添加动作滤波")
        elif jerk > 0.5 and jerk > 0.32 * 2:
            print("  建议: Jerk值比离线测试高，可能存在观测噪声或延迟问题")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ROS环境深入诊断工具")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/train/task1/act/run_20260202_223813/epochbest",
        help="模型路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deploy/kuavo_env.yaml",
        help="配置文件路径"
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
        help="诊断步数"
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
        default="outputs/diagnosis",
        help="输出目录"
    )
    parser.add_argument(
        "--save_actions",
        action="store_true",
        help="保存action序列"
    )
    parser.add_argument(
        "--skip_ros",
        action="store_true",
        help="跳过ROS初始化（用于测试脚本）"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    log.info(f"使用设备: {device}")
    
    # 检查ROS
    if not args.skip_ros:
        if not ROS_AVAILABLE:
            log.error("ROS不可用，请在ROS环境中运行此脚本")
            log.info("提示: 可以使用 --skip_ros 参数跳过ROS初始化进行脚本测试")
            sys.exit(1)
        
        # 初始化ROS节点
        try:
            rospy.init_node('ros_diagnostics', anonymous=True)
            log.info("ROS节点初始化成功")
        except Exception as e:
            log.error(f"ROS节点初始化失败: {e}")
            log.info("请确保ROS master已启动: roscore")
            sys.exit(1)
    
    # 加载配置
    log.info(f"加载配置: {args.config}")
    try:
        from kuavo_deploy.config import KuavoConfig
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        
        config_path = Path(args.config).resolve()
        initialize_config_dir(config_dir=str(config_path.parent), version_base=None)
        cfg = compose(config_name=config_path.stem)
        config = KuavoConfig(**OmegaConf.to_container(cfg, resolve=True))
        log.info("配置加载成功")
    except Exception as e:
        log.warning(f"使用Hydra加载配置失败: {e}, 尝试直接加载YAML")
        import yaml
        with open(args.config, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        # 创建简单的配置对象
        class SimpleConfig:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, SimpleConfig(v))
                    else:
                        setattr(self, k, v)
        config = SimpleConfig(cfg_dict.get('env', cfg_dict))
    
    # 获取配置参数 - 兼容 KuavoConfig (有.env属性) 和直接的env配置对象
    env_cfg = config.env if hasattr(config, 'env') else config
    ros_rate = getattr(env_cfg, 'ros_rate', 10)
    frame_alignment = getattr(env_cfg, 'frame_alignment', False)
    ratio = getattr(env_cfg, 'ratio', 1.0)
    
    log.info(f"配置: ros_rate={ros_rate}, frame_alignment={frame_alignment}, ratio={ratio}")
    
    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        log.error(f"模型路径不存在: {model_path}")
        sys.exit(1)
    
    if model_path.name.startswith("epoch"):
        pretrained_path = model_path.parent
    else:
        pretrained_path = model_path
    
    # 加载模型
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
    
    # 创建ObsBuffer
    if not args.skip_ros:
        log.info("创建ObsBuffer...")
        try:
            from kuavo_deploy.utils.obs_buffer import ObsBuffer
            # 启用内置计时功能
            obs_buffer = ObsBuffer(config, enable_timing=True)
            diag_obs_buffer = DiagnosticObsBuffer(obs_buffer)
            
            log.info("等待ObsBuffer就绪...")
            obs_buffer.wait_buffer_ready()
            log.info("ObsBuffer就绪")
        except Exception as e:
            log.error(f"创建ObsBuffer失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        log.info("跳过ROS，使用模拟数据...")
        diag_obs_buffer = None
    
    # 运行端到端诊断
    if diag_obs_buffer is not None:
        log.info("开始端到端诊断...")
        e2e_stats, actions = run_e2e_diagnosis(
            env=None,  # 不执行实际动作
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            diag_obs_buffer=diag_obs_buffer,
            num_steps=args.num_steps,
            device=device,
            frame_alignment=frame_alignment,
            ros_rate=ros_rate,
            ratio=ratio,
        )
        
        # 计算action平滑度
        action_smoothness = compute_action_smoothness(actions, dt=1.0/ros_rate)
        
        # 打印诊断报告
        print_diagnosis_report(
            diag_obs_buffer, e2e_stats, action_smoothness, actions, ros_rate
        )
        
        # 保存结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_actions:
            actions_path = output_dir / "diagnosis_actions.npy"
            np.save(actions_path, actions)
            log.info(f"Action序列已保存到: {actions_path}")
        
        # 保存统计数据
        stats_path = output_dir / "diagnosis_stats.json"
        stats_data = {
            "obs_buffer": {
                "get_obs": diag_obs_buffer.get_obs_times.get_stats(),
                "get_latest_obs": diag_obs_buffer.get_latest_obs_times.get_stats(),
                "get_aligned_obs": diag_obs_buffer.get_aligned_obs_times.get_stats(),
            },
            "e2e": {k: v.get_stats() for k, v in e2e_stats.items()},
            "action_smoothness": action_smoothness,
            "config": {
                "ros_rate": ros_rate,
                "frame_alignment": frame_alignment,
                "num_steps": args.num_steps,
            }
        }
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        log.info(f"诊断统计已保存到: {stats_path}")
        
        # 停止订阅
        obs_buffer.stop_subscribers()
    else:
        log.info("ROS跳过，仅测试模型加载和脚本逻辑")
        log.info("脚本测试成功，请在ROS环境中运行以获取完整诊断")
    
    log.info("诊断完成!")


if __name__ == "__main__":
    main()
