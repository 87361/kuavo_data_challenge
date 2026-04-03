# 🚀 **Kuavo Data Challenge**

<p align="right">
  <a href="README_ZH.md"><b>Chinese</b></a> |
  <a href="README.md">English</a>
</p>

[![leju](https://img.shields.io/badge/Leju Intelligent-blue)](https://www.lejurobot.com/zh)
[![tong](https://img.shields.io/badge/Beijing Institute of General Artificial Intelligence-red)](https://www.bigai.ai/)

---

## 🌟Project Introduction
This warehouse is developed based on [Lerobot](https://github.com/huggingface/lerobot) and combined with Leju Kuavo robot to provide complete sample code for **data format conversion** (rosbag → parquet), **imitation learning (IL) training**, **emulator testing** and **real machine deployment verification**.
 

---

## ✨ Core Functions
- Data format conversion module (rosbag → Lerobot parquet)
- IL model training framework (diffusion policy, ACT)
- Mujoco emulator support
- Real machine verification and deployment

⚠️ Note: This sample code does not yet support end control, currently only supports joint angle control!

---

## ♻️ Environmental requirements
- **System**: Ubuntu 20.04 recommended (22.04 / 24.04 recommended to run with Docker container)
- **Python**: Python 3.10 recommended
- **ROS**: ROS Noetic + Kuavo Robot ROS patch (supports installation within Docker)
- **Dependencies**: Docker, NVIDIA CUDA Toolkit (if GPU acceleration is required)

---

## 📦 Installation Guide

### 1. Operating system environment configuration
Recommended **Ubuntu 20.04 + NVIDIA CUDA Toolkit + Docker**.
<details>
<summary>Detailed steps (expand to view), for reference only</summary>

#### a. Install operating system and NVIDIA driver
```bash
sudo apt update
sudo apt upgrade -y
ubuntu-drivers devices
# The tested version is 535, you can try to update the version (do not use the server branch)
sudo apt install nvidia-driver-535
# Restart the computer
sudo reboot
# Verify driver
nvidia-smi
```

#### b. Install NVIDIA Container Toolkit

When using nvidia-smi acceleration in a docker image, the nvidia runtime library needs to be loaded, so the NVIDIA Container Toolkit needs to be installed.

```bash
sudo apt install curl
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1 && sudo apt-get install -y nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```


#### c. Install Docker

```bash
sudo apt update
sudo apt install git
sudo apt install docker.io
# Configure NVIDIA Runtime in docker
nvidia-ctk
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker info | grep -i runtime
# The output should contain "nvidia" Runtime
```

</details>

---

### 2. ROS environment configuration

Both the simulation and real-machine operation of kuavo mujoco are based on the **ROS Noetic** environment. Since the real-machine kuavo robot is ubuntu20.04 + ROS Noetic (non-docker), it is recommended to install ROS Noetic directly. If you cannot install ROS Noetic because the ubuntu version is higher, you can use docker.

<details>
<summary>a. Install ROS Noetic directly on the system (<b>recommended</b>)</summary>

* Official guide: [ROS Noetic Installation](http://wiki.ros.org/noetic/Installation/Ubuntu)
* Recommended domestic acceleration source: [FishROS](https://fishros.org.cn/forum/topic/20/)

Installation example:

```bash
wget http://fishros.com/install -O fishros && . fishros
# Menu selection: 5 Configure system source → 2 Change source and clean up third-party source → 1 Add ROS source
wget http://fishros.com/install -O fishros && . fishros
# Menu selection: 1 One-click installation → 2 Installation without changing the source → Select ROS1 Noetic desktop version
```

Test ROS installation:

```bash
roscore # Create a new terminal
rosrun turtlesim turtlesim_node # Create a new terminal
rosrun turtlesim turtle_teleop_key # Create a new terminal
```

</details>

<details>
<summary>b. Install ROS Noetic using Docker</summary>

- First it is best to change the source:

```bash
sudo vim /etc/docker/daemon.json
```

- Then write some mirror sources in this json file:

```json
{
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://docker.imgdb.de",
        "https://docker-0.unsee.tech",
        "https://docker.hlmirror.com",
        "https://docker.1ms.run",
        "https://func.ink",
        "https://lispy.org",
        "https://docker.xiaogenban1993.com"
    ]
}
```

- Then save the file and exit, restart the docker service:

```shell
sudo systemctl daemon-reload && sudo systemctl restart docker
```

- Now start creating the image, first create the Dockerfile:
```shell
mkdir /path/to/save/docker/ros/image
cd /path/to/save/docker/ros/image
vim Dockerfile
```
Then write the following content in the Dockerfile:

```Dockerfile
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y locales tzdata gnupg lsb-release
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

# Set the debian source of ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Add ROS Keys
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Install ROS Noetic
#Set the keyboard layout to Chinese
RUN apt-get update && \
    apt-get install -y keyboard-configuration apt-utils && \
    echo 'keyboard-configuration keyboard-configuration/layoutcode string cn' | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/modelcode string pc105' | debconf-set-selections && \
    echo 'keyboard-configuration keyboard-configuration/variant string ' | debconf-set-selections && \
    apt-get install -y ros-noetic-desktop-full && \
    apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    rm -rf /var/lib/apt/lists/*

#Initialize rosdep
RUN rosdep init
```
Save and exit after writing. Execute the build of ubuntu20.04 + ROS Noetic image:

```shell
sudo docker build -t ubt2004_ros_noetic .
```

After the build is completed, just enter the image and start the container for the first time to load the image:

```shell
sudo docker run -it --name ubuntu_ros_container ubt2004_ros_noetic /bin/bash
# or GPU startup (recommended)
sudo docker run -it --gpus all --runtime nvidia --name ubuntu_ros_container ubt2004_ros_noetic /bin/bash
# Optional, mount local directory path, etc.
# sudo docker run -it --gpus all --runtime nvidia --name ubuntu_ros_container -v /path/to/your/code:/root/code ubt2004_ros_noetic /bin/bash
```

Each subsequent load:
```shell
sudo docker start ubuntu_ros_container
sudo docker exec -it ubuntu_ros_container /bin/bash
```

After entering the image, initialize the ros environment variable and then start roscore

```shell
source /opt/ros/noetic/setup.bash
roscore
```

If it is correct, the docker configuration method of ubuntu20.04 + ros noetic is over.

</details>

<br>
⚠️ Warning: If the above ROS uses a docker environment, the subsequent code below may need to be run in the container. If there is any problem, please check whether it is currently in the container!

---

### 3. Clone code

```bash
# SSH
git clone --depth=1 git@github.com:LejuRobotics/kuavo_data_challenge.git
# or
# HTTPS
git clone --depth=1 https://github.com/LejuRobotics/kuavo_data_challenge.git
```

Update the lerobot submodule under third_party:

```bash
cd kuavo_data_challenge
git submodule init
git submodule update --recursive
```

---

### 4. Python environment configuration

Create a virtual environment using conda (recommended) or python venv (python 3.10 recommended):

```bash
conda create -n kdc python=3.10
conda activate kdc
```

Or: install python3.10 first, and then use venv to create a virtual environment

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev

python3.10 -m venv kdc
source kdc/bin/activate
```

Check to make sure the installation is correct:
```shell
python # Check the python version and see that the confirmation output is 3.10.xxx (usually 3.10.18)
# Output example:
# Python 3.10.18 (main, Jun  5 2025, 13:14:17) [GCC 11.2.0] on linux
# Type "help", "copyright", "credits" or "license" for more information.
# >>> 

pip --version # Check the version corresponding to pip and see the pip that confirms the output is 3.10
# Output example: pip 25.1 from /path/to/your/env/python3.10/site-packages/pip (python 3.10)
```


Install dependencies:

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple # It is recommended to change the source first to speed up download and installation.

pip install -r requirements_ilcode.txt # No need for ROS Noetic, but you can only use kuavo_train to imitate the learning training code. kuavo_data (digital transfer) and kuavo_deploy (deployment code) both rely on ROS
# or
pip install -r requirements_total.txt # Make sure ROS Noetic is installed (recommended)
```

If you get an error from ffmpeg or torchcodec when running:

```bash
conda install ffmpeg==6.1.1

# or

pip uninstall torchcodec
```

---

## 📨 How to use

### 1. Data format conversion

Convert Kuavo native rosbag data to parquet format usable by Lerobot framework:

```bash
python kuavo_data/CvtRosbag2Lerobot.py \
  --config-path=../configs/data/ \
  --config-name=KuavoRosbag2Lerobot.yaml \
  rosbag.rosbag_dir=/path/to/rosbag \
  rosbag.lerobot_dir=/path/to/lerobot_data
```

Description:

* `rosbag.rosbag_dir`: original rosbag data path
* `rosbag.lerobot_dir`: The converted lerobot-parquet data saving path, usually a subfolder named lerobot will be created in this directory
* `configs/data/KuavoRosbag2Lerobot.yaml`: Please view and select the enabled camera and whether to use depth images as needed

---

### 2. Imitation learning training

Use the converted data for imitation learning training:

```bash
python kuavo_train/train_policy.py \
  --config-path=../configs/policy/ \
  --config-name=diffusion_config.yaml \
  task=your_task_name \
  method=your_method_name \
  root=/path/to/lerobot_data/lerobot \
  training.batch_size=128 \
  policy_name=diffusion
```

Description:

* `task`: customized, task name (preferably corresponding to the task definition in the number transfer), such as `pick and place`
* `method`: custom, method name, used to distinguish different trainings, such as `diffusion_bs128_usedepth_nofuse`, etc.
* `root`: The local path of the training data. Note that lerobot is added. It needs to correspond to the data transfer saving path in 1, which is: `/path/to/lerobot_data/lerobot`
* `training.batch_size`: Batch size, can be adjusted according to GPU memory
* `policy_name`: The policy used, used for policy instantiation, currently supports `diffusion` and `act`
* For other parameters, please refer to the yaml file description. It is recommended to modify the yaml file directly to avoid command line input errors.

---

### 3. Emulator test

After completing the training, you can start the mujoco emulator and call the deployment code and evaluate:

a. Start the mujoco simulator: For details, please see [readme for simulator](https://github.com/LejuRobotics/kuavo-ros-opensource/blob/opensource/kuavo-data-challenge/readme.md)

b. Call deployment code

- Configuration files are located at `./configs/deploy/`:
  * `kuavo_sim_env.yaml`: emulator running configuration
  * `kuavo_real_env.yaml`: real machine running configuration


- Please check the yaml file and modify the following `# inference configs` related parameters (model loading), etc.

- Start automated inference deployment:
  ```bash
  bash kuavo_deploy/eval_kuavo.sh
  ```
- Follow the instructions. Generally, please select `" at the end. 8. Automatically test the model in simulation and execute eval_episodes times:`. For details on this step, please see [kuavo deploy](kuavo_deploy/readme/inference.md)
---



### 4. Real machine test

The steps are the same as part a in 3. Change the specified configuration file to `kuavo_real_env.yaml` to deploy the test on the real machine.

---

## 📡 ROS topic description

**Simulation environment:**

| Topic name | Function description |
| --------------------------------------------- | ------------- |
| `/cam_h/color/image_raw/compressed` | Upper camera RGB color image |
| `/cam_h/depth/image_raw/compressedDepth` | Top camera depth map |
| `/cam_l/color/image_raw/compressed` | Left camera RGB color image |
| `/cam_l/depth/image_rect_raw/compressedDepth` | Left camera depth map |
| `/cam_r/color/image_raw/compressed` | Right camera RGB color image |
| `/cam_r/depth/image_rect_raw/compressedDepth` | Right camera depth map |
| `/gripper/command` | Simulation rq2f85 gripper control command |
| `/gripper/state` | Current state of simulated rq2f85 gripper |
| `/joint_cmd` | Control instructions for all joints, including legs |
| `/kuavo_arm_traj` | Robot manipulator trajectory control |
| `/sensors_data_raw` | All sensor raw data |

**Real machine environment:**

| Topic name | Function description |
| --------------------------------------------- | ------------- |
| `/cam_h/color/image_raw/compressed` | Upper camera RGB color image |
| `/cam_h/depth/image_raw/compressedDepth` | Top camera depth map, realsense |
| `/cam_l/color/image_raw/compressed` | Left camera RGB color image |
| `/cam_l/depth/image_rect_raw/compressedDepth` | Left camera depth map, realsense |
| `/cam_r/color/image_raw/compressed` | Right camera RGB color image |
| `/cam_r/depth/image_rect_raw/compressedDepth` | Right camera depth map, realsense |
| `/control_robot_hand_position` | Dexterous hand joint angle control command |
| `/dexhand/state` | The current joint angle state of the dexterous hand |
| `/leju_claw_state` | Current joint angle status of Leju clamper |
| `/leju_claw_command` | Leju clamp joint angle control command |
| `/joint_cmd` | Control instructions for all joints, including legs |
| `/kuavo_arm_traj` | Robot manipulator trajectory control |
| `/sensors_data_raw` | All sensor raw data |



---

## 📁 Code output structure

```
outputs/
├── train/<task>/<method>/run_<timestamp>/ # Training model and parameters
├── eval/<task>/<method>/run_<timestamp>/ # Test log and video
```

---

## 📂 Core code structure

```
KUAVO_DATA_CHALLENGE/
├── configs/ # Configuration file
├── kuavo_data/ # Data processing conversion module
├── kuavo_deploy/ # Deployment script (simulator/real machine)
├── kuavo_train/ # Imitation learning training code
├── lerobot_patches/ # Lerobot running patches
├── outputs/ # Model and results
├── third_party/ # Lerobot dependency
└── requirements_xxx.txt # Dependency list
└── README.md # Documentation
```

---

## 🐒 About `lerobot_patches`

This directory contains compatibility patches for **Lerobot**. Its main features include:

* Extend `FeatureType` to support RGB and Depth images
* Customize `compute_episode_stats` and `create_stats_buffers`, used for statistics of image and depth data, min, max, mean, std, etc.
* Modify `dataset_to_policy_features` to ensure that the FeatureType of Kuavo RGB + Depth is correctly mapped

If you need to use customized designs based on lerobot, such as depth data, new FeatureType, normalization methods, etc., you can add them yourself and introduce them in the first line of the entry script (such as kuavo_train/train_policy.py and other training file codes) when using:

```python
import lerobot_patches.custom_patches  # Ensure custom patches are applied, DON'T REMOVE THIS LINE!
```

---

## 🙏 Acknowledgments

This project is based on [**Lerobot**](https://github.com/huggingface/lerobot).
Thanks to the HuggingFace team for developing the open source robot learning framework, which provides an important foundation for this project.


