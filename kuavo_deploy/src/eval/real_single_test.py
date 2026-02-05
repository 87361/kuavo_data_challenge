# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run kuavo_train/train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from lerobot_patches import custom_patches

from pathlib import Path

from sympy import im
from dataclasses import dataclass, field
import hydra
import gymnasium as gym
import imageio
import numpy
import torch
from tqdm import tqdm

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper
from kuavo_train.wrapper.policy.pi05.PI05PolicyWrapper import CustomPI05PolicyWrapper
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.random_utils import set_seed
import datetime
import time
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision.transforms.functional import to_tensor
from std_msgs.msg import Bool
import rospy
import threading

from kuavo_deploy.config import KuavoConfig, VLASHOptimizationConfig
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.kuavo_service.client import PolicyClient
from kuavo_deploy.utils.inference_utils import warmup_policy, compile_policy
from kuavo_deploy.utils.async_manager import AsyncInferenceManager, SyncInferenceManager
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.policies.factory import make_pre_post_processors

log_model = setup_logger("model")
log_robot = setup_logger("robot")

def pause_callback(msg):
    if msg.data:
        pause_flag.set()
    else:
        pause_flag.clear()

def stop_callback(msg):
    if msg.data:
        stop_flag.set()

pause_sub = rospy.Subscriber('/kuavo/pause_state', Bool, pause_callback, queue_size=10)
stop_sub = rospy.Subscriber('/kuavo/stop_state', Bool, stop_callback, queue_size=10)
stop_flag = threading.Event()
pause_flag = threading.Event()


def setup_policy(pretrained_path, policy_type, device=torch.device("cuda"), 
                 vlash_config: VLASHOptimizationConfig = None):
    """
    Set up and load the policy model.
    
    Args:
        pretrained_path: Path to the checkpoint
        policy_type: Type of policy ('diffusion' or 'act')
        device: Target device for inference
        vlash_config: VLASH optimization configuration (optional)
        
    Returns:
        Loaded policy model
    """
    
    if device.type == 'cpu':
        log_model.warning("Warning: Using CPU for inference, this may be slow.")
        time.sleep(3)  
    
    if policy_type == 'diffusion':
        policy = CustomDiffusionPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type == 'act':
        policy = CustomACTPolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type == 'pi05':
        policy = CustomPI05PolicyWrapper.from_pretrained(Path(pretrained_path),strict=True)
    elif policy_type == 'client':
        policy = PolicyClient()
    else:
        raise ValueError(f"Unsupported policy type: {policy_type}")
    
    policy.eval()
    policy.to(device)
    policy.reset()
    
    # Log model info
    log_model.info(f"Model loaded from {pretrained_path}")
    log_model.info(f"Model n_obs_steps: {policy.config.n_obs_steps}")
    log_model.info(f"Model device: {device}")
    
    # Apply VLASH optimizations if configured
    if vlash_config is not None and vlash_config.use_torch_compile:
        log_model.info(f"Applying torch.compile optimization (mode: {vlash_config.compile_mode})...")
        try:
            torch.set_float32_matmul_precision("high")
            policy = torch.compile(policy, mode=vlash_config.compile_mode)
            log_model.info(f"torch.compile applied successfully")
            
            # Warmup the compiled policy
            if vlash_config.warmup_steps > 0:
                log_model.info(f"Warming up compiled policy ({vlash_config.warmup_steps} steps)...")
                warmup_time = warmup_policy(policy, device, vlash_config.warmup_steps)
                log_model.info(f"Warmup completed in {warmup_time:.2f}s")
        except Exception as e:
            log_model.warning(f"torch.compile failed: {e}. Using original policy.")
    
    return policy

def main(config: KuavoConfig, env: gym.Env):
    # load config
    cfg = config.inference

    eval_episodes = cfg.eval_episodes
    seed = cfg.seed
    start_seed = cfg.start_seed
    policy_type = cfg.policy_type
    task = cfg.task
    method = cfg.method
    timestamp = cfg.timestamp
    epoch = cfg.epoch
    env_name = cfg.env_name

    pretrained_path = Path(f"outputs/train/{task}/{method}/{timestamp}/epoch{epoch}")
    output_directory = Path(f"outputs/eval/{task}/{method}/{timestamp}/epoch{epoch}")
    # Create a directory to store the video of the evaluation
    output_directory.mkdir(parents=True, exist_ok=True)

    # set seed
    set_seed(seed=seed)

    # Select your device
    device = torch.device(cfg.device)

    # Get VLASH optimization config
    vlash_config = cfg.vlash_optimization
    log_model.info(f"VLASH optimization: torch_compile={vlash_config.use_torch_compile}, async_inference={vlash_config.use_async_inference}")

    policy = setup_policy(pretrained_path, policy_type, device, vlash_config)
    # preprocessor = PolicyProcessorPipeline.from_pretrained(pretrained_path, config_filename="policy_preprocessor.json")
    # postprocessor = PolicyProcessorPipeline.from_pretrained(pretrained_path, config_filename="policy_postprocessor.json")
    preprocessor, postprocessor = make_pre_post_processors(None, Path(str(pretrained_path).split("/epoch", 1)[0]))

    # Initialize inference manager based on VLASH config
    inference_manager = None
    if vlash_config.use_async_inference:
        # Determine n_action_steps: use override if provided, otherwise use model config
        n_action_steps = vlash_config.n_action_steps_override
        if n_action_steps <= 0:
            # Try to get from policy config (chunk_size for ACT)
            if hasattr(policy, 'config'):
                if hasattr(policy.config, 'chunk_size'):
                    n_action_steps = policy.config.chunk_size
                elif hasattr(policy.config, 'n_action_steps'):
                    n_action_steps = policy.config.n_action_steps
                else:
                    n_action_steps = 20  # Default fallback
            else:
                n_action_steps = 20
        
        inference_manager = AsyncInferenceManager(
            policy=policy,
            n_action_steps=n_action_steps,
            overlap_steps=vlash_config.inference_overlap_steps,
            device=device,
            use_future_state=vlash_config.use_future_state_awareness,
        )
        log_model.info(f"Async inference enabled: n_action_steps={n_action_steps}, "
                       f"overlap_steps={vlash_config.inference_overlap_steps}, "
                       f"future_state={vlash_config.use_future_state_awareness}")

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent.
    max_episode_steps = cfg.max_episode_steps
    env = env

    # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # produced by the environment
    if policy_type != 'client':
        log_model.info(f"policy.config.input_features: {policy.config.input_features}")
        log_robot.info(f"env.observation_space: {env.observation_space}")

    # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # environment
    if policy_type != 'client':
        log_model.info(f"policy.config.output_features: {policy.config.output_features}")
        log_robot.info(f"env.action_space: {env.action_space}")

    # Log evaluation results
    log_file_path = output_directory / "evaluation.log"
    with log_file_path.open("w") as log_file:
        log_file.write(f"Evaluation Timestamp: {datetime.datetime.now()}\n")
        log_file.write(f"Total Episodes: {eval_episodes}\n")

    success_count = 0
    for episode in tqdm(range(eval_episodes), desc="Evaluating model", unit="episode"):
        # Reset the policy and environments to prepare for rollout
        policy.reset()
        if inference_manager is not None:
            inference_manager.reset()
        observation, info = env.reset(seed=episode+start_seed)
        observation = preprocessor(observation)
        # log_file.write(f"~~~~~~~~~~~~~~~~~~preprocess observation ok!~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        # Prepare to collect every rewards and all the frames of the episode,
        # from initial state to final state.
        rewards = []

        cam_keys = [k for k in observation.keys() if "images" in k or "depth" in k]
        frame_map = {k: [] for k in cam_keys}

        average_exec_time = 0
        average_action_infer_time = 0
        average_step_time = 0

        step = 0
        done = False
        with tqdm(total=max_episode_steps, desc=f"Episode {episode+1}", unit="step", leave=False) as pbar:
            while not done:
                # --- Pause support: block here if pause_flag is set ---
                while pause_flag.is_set() and not stop_flag.is_set():
                    log_model.info("Paused. Waiting for resume signal...")
                    time.sleep(0.5)
                if stop_flag.is_set():
                    log_model.info("Stop flag detected during pause. Exiting loop.")
                    return
                
                start_time = time.time()
                
                # Use inference manager if available, otherwise use original logic
                if inference_manager is not None:
                    # Async inference: manager handles chunking and timing
                    action = inference_manager.get_action(observation, None, postprocessor)
                else:
                    # Original synchronous inference
                    with torch.inference_mode():
                        action = policy.select_action(observation)
                    action = postprocessor(action)
                
                action_infer_time = time.time()
                log_model.debug(f"action infer time: {action_infer_time - start_time:.3f}s")
                average_action_infer_time += action_infer_time - start_time

                numpy_action = action.squeeze(0).cpu().numpy()
                log_model.debug(f"numpy_action: {numpy_action}")

                # 执行动作 Execute action
                observation, reward, terminated, truncated, info = env.step(numpy_action)
                observation = preprocessor(observation)
                exec_time = time.time()
                log_model.debug(f"exec time: {exec_time - action_infer_time:.3f}s")
                average_exec_time += exec_time - action_infer_time

                rewards.append(reward)

                # 相机帧记录，真机请取消，否则会一直堆叠卡死
                # Camera frame record, must be commented out during real-device testing

                # for k in cam_keys:
                #     frame_map[k].append(observation[k].squeeze(0).cpu().numpy().transpose(1, 2, 0))

                # The rollout is considered done when the success state is reached (i.e. terminated is True),
                # or the maximum number of iterations is reached (i.e. truncated is True)
                done = terminated | truncated | done
                step += 1

                end_time = time.time()
                log_model.info(f"Step {step} time: {end_time - start_time:.3f}s")
                
                # Update progress bar
                status = "Success" if terminated else "Running"
                pbar.set_postfix({
                    "Reward": f"{reward:.3f}",
                    "Status": status,
                    "Total Reward": f"{sum(rewards):.3f}"
                })
                pbar.update(1)

        if terminated:
            success_count += 1
            log_model.info(f"✅ Episode {episode+1}: Success! Total reward: {sum(rewards):.3f}")
        else:
            log_model.info(f"❌ Episode {episode+1}: Failed! Total reward: {sum(rewards):.3f}")

        # Get the speed of environment (i.e. its number of frames per second).
        fps = env.ros_rate

        log_model.info(f"average exec time: {average_exec_time / step:.3f}s")
        log_model.info(f"average action infer time: {average_action_infer_time / step:.3f}s")
        log_model.info(f"average step time: {average_step_time / step:.3f}s")
        log_model.info(f"average sleep time: {env.average_sleep_time / step:.3f}s")
        
        # Log inference manager statistics if using async mode
        if inference_manager is not None:
            stats = inference_manager.get_statistics()
            log_model.info(f"[Async Inference Stats] total_inferences={stats['total_inferences']}, "
                           f"mean_time={stats['mean_inference_time']:.3f}s, "
                           f"min_time={stats['min_inference_time']:.3f}s, "
                           f"max_time={stats['max_inference_time']:.3f}s")
        
        
        # Encode all frames into a mp4 video.
        if len(frame_map.keys()) == 0:
            for cam in cam_keys:
                frames = frame_map[cam]
                output_path = output_directory / f"rollout_{episode}_{cam}.mp4"
                imageio.mimsave(str(output_path), frames, fps=fps)

        # print(f"Video of the evaluation is available in '{video_path}'.")

        with log_file_path.open("a") as log_file:
            log_file.write("\n")
            log_file.write(f"Rewards per Episode: {numpy.array(rewards).sum()}")

    with log_file_path.open("a") as log_file:
        log_file.write("\n")
        log_file.write(f"Success Count: {success_count}\n")
        log_file.write(f"Success Rate: {success_count / eval_episodes:.2f}\n")

    # Display final statistics
    log_model.info("\n" + "="*50)
    log_model.info(f"🎯 Evaluation completed!")
    log_model.info(f"📊 Success count: {success_count}/{eval_episodes}")
    log_model.info(f"📈 Success rate: {success_count / eval_episodes:.2%}")
    print(f"📁 Videos and logs saved to: {output_directory}")
    print("="*50)

def kuavo_eval(config: KuavoConfig, env: gym.Env):
    main(config, env)

if __name__ == "__main__":
    config_path = Path("test.yaml")
    env = gym.make(
        "Kuavo-Real",
        max_episode_steps=150,
        config_path=config_path,
    )
