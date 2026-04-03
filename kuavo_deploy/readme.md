# 🤖 Kuavo Deploy - robot deployment module

> The ROS-based Kuavo robot deployment module supports model reasoning, trajectory playback, and robotic arm control for real robots and simulation environments.

## 📁 Module structure

```
kuavo_deploy/
├── kuavo_env/ # Robot environment module
│ ├── kuavo_real_env/ # Real robot environment
│ │ └── KuavoRealEnv.py # Real robot environment implementation
│ ├── kuavo_sim_env/ # Simulation environment
│ │ └── KuavoSimEnv.py # Simulation environment implementation
│ └── KuavoBaseRosEnv.py # ROS environment base class
├── examples/ # Example code and evaluation
│ ├── eval/ # Evaluation script
│ │ ├── eval_kuavo.py # Kuavo environment assessment script
│ │ └── auto_test/ # Automated testing
│ │ ├── eval_kuavo.py # Kuavo environment automation evaluation script
│ │ └── eval_kuavo_autotest.py # Automated test script
│ └── scripts/ # Control script
│ ├── script.py # Main control script
│ ├── controller.py # Robotic arm controller
│ └── script_auto_test.py # Automation control script
├── utils/ # tool module
│ └── logging_utils.py # Logging tool
└── eval_others.py # Other environments such as pusht, aloha and other evaluation scripts

configs/
└── deploy/ # Deployment configuration (new location)
    ├── config_inference.py # Inference configuration loader
    ├── config_kuavo_env.py # Environment configuration loader
    ├── kuavo_real_env.yaml # kuavo real environment parameters
    ├── kuavo_sim_env.yaml # kuavo simulation environment parameters
    └── others_env.yaml #Other evaluation environment parameters

log/
└── kuavo_deploy/                
    └── kuavo_deploy.log # Reasoning log

outputs/
└── eval/
    └── {task}/ #Task name (such as pick_place, push, etc.)
        └── {method}/ #Method name (such as diffusion, act, etc.)
            └── {timestamp}/ # Running timestamp
                └── epoch{epoch}/ # Model weight round used for evaluation
                    ├── evaluation.log # Manual evaluation log
                    ├── evaluation_autotest.log # Automated evaluation log
                    ├── evaluation_autotest.json # Automated evaluation results json
                    ├── rollout_0_observation.images.head_cam_h.mp4 # episode0 head camera video
                    ├── rollout_0_observation.images.wrist_cam_l.mp4 # episode0 left wrist camera video
                    ├── rollout_0_observation.images.wrist_cam_r.mp4 # episode0 right wrist camera video
                    ├── rollout_1_observation.images.head_cam_h.mp4 # episode1 head camera video
                    ├── rollout_1_observation.images.wrist_cam_l.mp4 # episode1 Left wrist camera video
                    ├── rollout_1_observation.images.wrist_cam_r.mp4 # episode1 right wrist camera video
                    └── ... # and so on (rollout_2, rollout_3, ...)
```

## ✨ Main features

### 🎯 Core Functions
- **Real Robot Control**: Supports mechanical arm control of Kuavo real robots
- **Simulation Environment**: Provides a simulation environment for model testing and verification
- **Model Inference**: Supports real-time inference of Diffusion Policy and other models
- **Trajectory Playback**: Supports the trajectory playback function of ROS bag files
- **Multi-modal input**: Supports image input from head camera and wrist camera
- **End effector**: Supports two end effectors: strong brain and dexterous hand and gripper

### 🔧 Technical characteristics
- **ROS Integration**: ROS-based robot control architecture
- **Gymnasium Environment**: Standardized reinforcement learning environment interface
- **Real-time control**: Supports 10Hz real-time inference frequency
- **SAFETY MECHANISMS**: Built-in joint limits and safety checks
- **Log System**: Complete logging and debugging functions

### 🎮 Control Mode
- **Joint Control**: Directly control the joint angle of the robotic arm
- **Trajectory Interpolation**: Smooth trajectory interpolation algorithm
- **Emergency Stop**: Supports interrupt and emergency stop functions

## 🚀 Quick Start

### 1. Pull the code repository from github

```bash
git clone https://github.com/LejuRobotics/kuavo_data_challenge.git
cd kuavo_data_challenge
git submodule update --init --recursive
```

### 2. Configuration environment (ros, conda, python environment)

Refer to [Environment Configuration Guide](readme/setup_env.md)

### 3. Real machine wired communication configuration

Reference [Robot connection configuration](readme/setup_robot_connection.md)

### 4. Simulation and real machine reasoning

Reference [Inference Guide](readme/inference.md)
