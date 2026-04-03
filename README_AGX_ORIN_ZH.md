# Kuafu Robot Host Computer Reasoning Guide

## 1. Connect to the robot host computer

There are two ways to connect to the host computer:

(a) Use a computer on the same LAN as the robot and connect via SSH
```bash
ssh leju_kuavo@xxx.xxx.xxx.xxx # Confirm the host computer IP by yourself
# Password:leju_kuavo
````

(b) Use keyboard, mouse and monitor to directly connect to the robot host computer (the subsequent steps are the same)

---

## 2. Create a working directory and prepare the environment

Create working directory:

```bash
cd ~
mkdir kdc_ws
cd kdc_ws
```

Clone the code repository:

```bash
# use https
git clone https://github.com/LejuRobotics/kuavo_data_challenge.git

# Or use ssh
# git clone git@github.com:LejuRobotics/kuavo_data_challenge.git
```

Initialize branches and submodules:

```bash
cd kuavo_data_challenge
git checkout origin/dev
git submodule init
git submodule update --recursive --progress

# If this step fails to download or is very slow due to network reasons: please
# cd third_party
# git clone https://githubproxy.cc/https://github.com/huggingface/lerobot.git
# cd ../ # Return to the previous directory
```

---

## 3. Create a Python environment and install dependencies

Usually the host computer has Python 3.10 pre-installed. If it is not installed, please follow the appendix to complete the installation of Python 3.10 first.

```bash
python3.10 -m venv ~/kdc_ws/kdc_env
source ~/kdc_ws/kdc_env/bin/activate

which pip
pip list

# If you need ROS dependencies
# source /opt/ros/noetic/setup.bash

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements_agxorin.txt

# If dependency conflicts occur due to pip version issues:
pip install -r requirements_agxorin.txt --use-deprecated=legacy-resolver
```

---

## 4. Place the trained weights

Copy the complete directory of training output to the following path:

```
~/kdc_ws/kuavo_data_challenge/outputs/train/<task>/<method>/<timestamp>/epoch<epoch>
```

Example:

```bash
mkdir -p outputs/train/your_task_name/your_method/your_timestamp
cp -R <your_epoch_dir> outputs/train/your_task_name/your_method/your_timestamp
```

The directory structure should be as follows:

```
outputs
 └── train
     └── your_task_name
         └── your_method
             └── timestamp
                 ├── epochxxx
                 │   ├── config.json
                 │   └── model.safetensors
                 ├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
                 ├── policy_postprocessor.json
                 ├── policy_preprocessor_step_3_normalizer_processor.safetensors
                 └── policy_preprocessor.json
```

---

## 5. Configure and run inference

Edit deployment configuration:

```bash
vim configs/deploy/kuavo_env.yaml
```

Please be sure to confirm that the configuration is correct item by item, otherwise the reasoning may not work properly.
(vim: `ESC` → `:wq!` save and exit; `:q!` abandon modifications)

Start reasoning:

```bash
python kuavo_deploy/eval_kuavo.py
```

Enter in sequence according to the prompts:

(a) Enter **3**
(b) Enter `configs/deploy/kuavo_env.yaml`
(c) Enter **2** to play back rosbag (the robot will start to move, please pay attention to safety)

After playback ends:

(d) Enter **3** to start reasoning
(e) You can press **s** to stop at any time during the reasoning process (recommended), or `Ctrl+C` to end

---

## Appendix:

### python3.10 installation:

⚠️ Note: ```ppa:deadsnakes``` will no longer be available on ubuntu20.04 after June 2025. The following installation method may not be successful:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
```

You can try it. If it doesn’t work, please use the source code to install:

```bash
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-devlibreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev uuid-dev wget
wget https://www.python.org/ftp/python/3.10.18/Python-3.10.18.tgz
tar -xzf Python-3.10.18.tgz
cd Python-3.10.18
./configure --prefix=$HOME/python3.10 --enable-optimizations
make -j$(nproc)
sudo make install
```

---
### About kuavo_humanoid_sdk:

⚠️ Sometimes there will be version mismatch problems, unable to communicate or something, and an error will be reported: The robot arm initialization failed! Solution, if related problems occur:

(a) Enter the robot lower machine,

```bash
  ssh lab@192.168.26.1 # Password three spaces
  cd ~/kuavo-ros-opensource
  git describe --tag # View opensource version
  # show xxx
```
  - Return to the side machine or host machine,
```bash
# Enter the environment
conda activate kdc_dev
# or
source kdc_dev/bin/activate
pip install kuavo-humanoid-sdk==xxx #Install the corresponding version of sdk
```


(b) (It takes longer and is more complicated, not recommended) You can copy and install the content of kuavo-ros-opensource of the robot slave machine, [kuavo-ros-opensource](https://github.com/LejuRobotics/kuavo-ros-opensource), for example,

```bash
scp -r lab@192.168.26.1:~/kuavo-ros-opensource /your/path/
cd /your/path/kuavo-ros-opensource/src/kuavo_humanoid_sdk
# or
# cd /your/path/to/kuavo-ros-opensource/src/kuavo_humanoid_sdk
# Enter the environment
conda activate kdc_dev
# or
source kdc_dev/bin/activate

./install.sh
```
