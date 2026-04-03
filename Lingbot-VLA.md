# Lingbot environment configuration
1. Install the kdc environment
2. Refer to the README.md of https://github.com/Robbyant/lingbot-vla to download the lingbot-vla-4b, Qwen2.5-VL-3B-Instruct model
3. In the kdc environment, use the provided lingbot-vla warehouse and run pip install -e . and pip install -r requirements.txt
4. Install flash-attention in the kdc environment, pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
5. Enter the third_party/lerobot folder of kdc, pip install -e. Install lerobot
6. The side computer uses kdc version 1.3.3, pip install kuavo-humannoid-sdk==1.3.3, and the lower computer uses kuavo-ros-control version 1.3.3. You can use git reset --hard 1.3.3 and then use catkin build To compile humanoid_cotrollers, you need to modify src/kuavo_assets/config/kuavo_v49/kuavo.json to replace qiangnao with lejuclaw, otherwise /leju_claw_state cannot be monitored

# LingBot Training Instructions

This article explains how to start LingBot training in the current warehouse, and the actual relationship between the two layers of configuration files.

## 1. Data processing

Convert rosbag to Lerobot dataset:

```bash
python kuavo_data/CvtRosbag2Lerobot.py --config-path=../configs/data/ --config-name=KuavoRosbag2Lerobot.yaml rosbag.rosbag_dir=/home/lmy/lmy_ws/Leju/kuavo_data_challenge/raw_data rosbag.lerobot_dir=/home/lmy/lmy_ws/Leju/pick_and_place
```

Generate norm_stats configuration files based on data set transformation for training and inference:

`  python kuavo_train/convert_stats_to_norm_stats.py \
    /home/lmy/lmy_ws/pick_and_place/pick_and_place0_4_2/meta/stats.json`


## 1. Training entrance

Current training entry command:

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml
```

lingbot_config.yaml needs to be modified:

1. `root: /data/limingyang/VLA/huggingface/pick_and_place/pick_and_place0_4_2`

2. `model_path: /home/yunxi/lmy/VLA/lingbot-vla-4b`

3. `tokenizer_path: /home/yunxi/lmy/VLA/Qwen2.5-VL-3B-Instruct`

robotwin_load20000h.yaml needs to be modified:

1. `norm_stats_file: assets/norm_stats/pick_and_place0_4_2.json` needs to match the data set used


Tensorboard visualization:

```bash
tensorboard --logdir outputs/train/lingbot_task/lingbot_post_train --port 6007
```

This command will go to the `policy_name=lingbot` branch, call `torchrun` from `kuavo_train/train_policy.py`, and finally execute:

```text
kuavo_train/lingbot/tasks/vla/train_lingbotvla.py
```

## 2. Two-tier configuration file

Training actually consists of two layers of configuration:

### Outer startup configuration

File:

```text
configs/policy/lingbot_config.yaml
```

Function:

- Specify the training entrance to use `lingbot`
- Specify the data root directory `root`
- Specify the model path `policy.model_path`
- Specify tokenizer path `policy.tokenizer_path`
- Specify the graphics card `training.gpu_ids` / `policy.env.CUDA_VISIBLE_DEVICES`
- Specify the outer output directory template `training.output_directory`
- Specify whether to continue training `training.resume`

### Inner LingBot training configuration

File:

```text
configs/policy/lingbot/robotwin_load20000h.yaml
```

Function:

- Control LingBot training details
- Includes default values for `train.*`, `data.*`, `model.*`

## 3. Configure priority

The priority is not a simple choice between two, but:

1. First read `lingbot_config.yaml`
2. Read `robotwin_load20000h.yaml` again
3. The outer launcher converts some parameters into command line parameters
4. Command line parameters cover the corresponding fields in `robotwin_load20000h.yaml`

That is to say:

- **No parameters passed by the outer layer**, finally use `robotwin_load20000h.yaml`
- **Parameters explicitly passed by the outer layer**, whichever is the outer layer

The key parameters currently covered by the outer layer include:

- `data.train_path`
- `train.action_dim`
- `train.micro_batch_size`
- `train.global_batch_size`
- `train.output_dir`
- `model.model_path`
- `model.tokenizer_path`
- Optional `model.moge_path`
- Optional `model.morgbd_path`

So if you change many `training.*` fields in `lingbot_config.yaml`, but the launcher does not transparently pass it to LingBot, then these fields will not automatically overwrite the inner training configuration.

## 4. Current recommended training methods

### 4.1 Check the critical path first

Please first confirm that these paths exist:

```text
/home/yunxi/lmy/VLA/lingbot-vla
/home/yunxi/lmy/VLA/lingbot-vla-4b
/home/yunxi/lmy/VLA/Qwen2.5-VL-3B-Instruct
/home/yunxi/lmy/VLA/huggingface/lerobot/pick_and_place0_4_2
```

### 4.2 Start training directly

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml
```

### 4.3 Just look at what command will eventually be started

If you just want to check the final generated `torchrun` command without actually training:

```bash
python kuavo_train/train_policy.py --config-path=../configs/policy/ --config-name=lingbot_config.yaml policy.dry_run=true
```

## 5. Current configuration recommendations

### Graphics card

If you want to train multiple cards, it is recommended to set it directly in `configs/policy/lingbot_config.yaml`:

```yaml
policy:
  env:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

If `policy.env.CUDA_VISIBLE_DEVICES` is not set, the launcher will try to refer to `training.gpu_ids`.

### batch size

In the current outer configuration:

```yaml
training:
  batch_size: 1
```

will be converted to:

- `--train.micro_batch_size 1`
- `--train.global_batch_size = micro_batch_size * number of GPUs`

For example, when there are 8 cards, it will become:

```text
micro_batch_size=1
global_batch_size=8
```

## 6. Continue training

In the outer configuration:

```yaml
training:
  resume: true
  resume_timestamp: "20260320_063225"
```

When `resume=true` and `resume_timestamp` is not empty, the launcher will switch the output directory back to the old run directory.

If you do not continue training:

```yaml
training:
  resume: false
```

will generate a new one:

```text
outputs/train/${task}/${method}/run_${timestamp}
```

## 7. How to modify training logic

### Want to change training details

Please modify it first:

```text
configs/policy/lingbot/robotwin_load20000h.yaml
```

For example:

- learning rate
- epoch
- save_steps
- FSDP / offload
- tokenizer length
- Other `train.*` / `data.*` default behaviors

### Want to change the behavior of the entry layer

Please modify:

```text
configs/policy/lingbot_config.yaml
```

For example:

- Data root directory
- model path
- tokenizer path
- Graphics card
- Whether to dry run
- whether to resume
- Output directory template

## 8. Recommended usage habits

- Put the real LingBot training parameters in `robotwin_load20000h.yaml`
- Put the outer path, graphics card, resume, and output directory in `lingbot_config.yaml`
- After modification, run `policy.dry_run=true`
- Confirm that the generated `torchrun` command is correct before starting formal training

## 9. Real machine deployment and testing

### 9.1 Premise

By default you have completed:

- LingBot model training
- `configs/deploy/kuavo_env.yaml` has been changed to the deployment configuration corresponding to the current data set
- The current usage environment is `kdc_dev`
- The real machine side already has the `kuavo-ros-opensource` running conditions

The current deployment characteristics correspond to:

- `observation.images.head_cam_h`
- `observation.images.wrist_cam_r`
- `observation.state`
- `action`

That is, the configuration of "head camera + right wrist camera + single right arm status/action".

### 9.2 Robot and network preparation

After connecting to the `Kuavo-manipulation` network, you can view the current device IP through `http://192.168.5.1/`. Make sure that the upper and lower computers and side devices are in the same network segment and can communicate with each other.

Connect the robot slave machine:

```bash
ssh lab@192.168.5.117
```

After entering the control warehouse, start the motion control node:

```bash
cd kuavo-ros-control
sudo su
roslaunch humanoid_controllers load_kuavo_real.launch
```

Connect to the robot host computer:

```bash
ssh leju_kuavo@192.168.5.111
```

Start the camera:

```bash
sudo systemctl start start_camera.service
sudo systemctl restart start_camera.service
```

If the camera cannot start, first check whether the `ROS_MASTER_URI` in the `/etc/kuavo.conf` of the host computer points to the correct remote ROS Master, and be careful not to change `ROS_IP` by mistake.

If the side machine keeps prompting that the topic is empty when running the deployment script, but the network between the three machines is interconnected, you need to check whether the `kuavo_master` mapping has been added to the side machine `/etc/hosts`, for example:

```text
127.0.0.1 localhost
127.0.1.1 myl
192.168.5.165 kuavo_master
```
Note that if there is a message failure, check ~/.bashrc and /etc/hosts on the corresponding device.

If the side computer message definition is missing, you can also copy `kuavo-ros-control` directly from the lower computer.

### 9.3 Check before boarding the machine

It is recommended to confirm first:

- The current environment is `kdc_dev`
- `policy_type: lingbot` in `configs/deploy/kuavo_env.yaml`
- `go_bag_path` in `configs/deploy/kuavo_env.yaml` has been changed to the real absolute path
- `pretrained_path` in `configs/deploy/kuavo_env.yaml` points to the real LingBot `hf_ckpt`
- The real machine ROS topic and deployment configuration are consistent

You can check the model directory first:

```bash
ls /home/yunxi/lmy/VLA/kuavo_data_challenge/outputs/train/lingbot_task/lingbot_post_train/run_20260314_061741/checkpoints/global_step_15300/hf_ckpt
```

### 9.4 Check ROS observation link

Execute in a new terminal:

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
rostopic hz /cam_h/color/image_raw/compressed
rostopic hz /cam_r/color/image_raw/compressed
rostopic echo /sensors_data_raw
rostopic echo /leju_claw_state
```

At least confirm the existence of the following four types of observations:

- Head RGB
- Right wrist RGB
- joint status
- Gripper status

If these topics are inconsistent with `configs/deploy/kuavo_env.yaml`, change the configuration first and then continue deployment.

### 9.5 Real machine test sequence

It is recommended to proceed strictly in the following order:

1. Start the bottom layer of the real machine
2. Check out 4 key topics
3. Test `go`
4. Test `run`
5. Final test `go_run`

Don't run `go_run` directly at the beginning.

#### Phase 1: Only test `go.bag`

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py --task go --config configs/deploy/kuavo_env.lmy_go_run.yaml
```
Things that need to be modified in kuavo_env.lmy_go_run.yaml:
1. rosbag path:

`go_bag_path: /home/lmy/lmy_ws/Leju/kuavo_data_challenge_dev/raw_data/A01-A02-A02-A02-A02-A03-P4_000-leju_claw-20260309165457-v002.bag`

2. Pre-training model path:

`pretrained_path: "/home/lmy/lmy_ws/Leju/kuavo_data_challenge_master/models/run_20260323_060215_7005/run_20260323_060215/checkpoints/global_step_7005/hf_ckpt"`

3. lingbot-vla path:

`lingbot_root: "/home/lmy/lmy_ws/Leju/lingbot-vla"`

4. Inference chunk:

`lingbot_use_length: 5`

5. norm_stats file:

`lingbot_norm_stats_file: "/home/lmy/lmy_ws/Leju/kuavo_data_challenge_master/assets/norm_stats/pick_and_place0_4_2.json"`

6. If lingbot_use_length is greater than 1, lingbot_chunk_ret needs to be set to false

7. Qwen2.5 path:

`qwen25_path: "/home/lmy/lmy_ws/Leju/Qwen2.5-VL-3B-Instruct"`

Instructions through this step:

- `go_bag_path` is readable
- Track playback link is normal
- There are no obvious errors in the current right arm and gripper control mapping
- The starting position is at least reachable

If this step is unsafe, do not continue running the model.

#### Second stage: run the model directly from the current position

Give the path to QWEN2.5 in the terminal:

`export QWEN25_PATH=/home/yunxi/lmy/VLA/Qwen2.5-VL-3B-Instruct`

Manually place the robot in the safe starting position and then execute:

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py --task run --config configs/deploy/kuavo_env.lmy_go_run.yaml
```

This step mainly verifies:

- Can LingBot checkpoint be loaded?
- Can the head + right wrist + 8-dimensional state enter the model normally?
- Whether the model output action can be successfully sent to the real machine
- Is the direction of action basically reasonable?

Suggestions for first test on real machine:

- `eval_episodes=1`
- Someone is on duty nearby
- Be prepared for physical emergency stops at any time
- Only observe whether the first few steps are reasonable.

#### The third stage: complete `go_run`

After passing the first two steps, execute:

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py --task go_run --config configs/deploy/kuavo_env.lmy_go_run.yaml
```

This process will first press `go.bag` to the starting posture, and then start LingBot inference.

### 9.6 Pause, stop and return to zero

Header control can be issued via:

```bash
rostopic pub /robot_head_motion_data kuavo_msgs/robotHeadMotionData "joint_data: [0.0, 27.0]" --once
```

`script.py` will print the PID after startup, which can be controlled in the following ways:

Pause or resume:

```bash
kill -USR1 <pid>
```

Stop:

```bash
kill -USR2 <pid>
```

If you need to return to a safe position after stopping:

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py --task back_to_zero --config configs/deploy/kuavo_env.lmy_go_run.yaml
```

It is recommended to verify `back_to_zero` separately before the first real machine test.

### 9.7 FAQ

- `Robotic arm initialization failed`
  - Prioritize checking whether the version of `kuavo-humanoid-sdk` is consistent with the robot side.

- `LingBot import failed`
  - Confirm that the current environment is `kdc_dev`, and that LingBot related modules can be imported normally in this environment.

- `wrist_cam_l is missing`
  - The current warehouse is compatible with deployment scenarios that only have the right wrist image.

- The model can be started but behaves abnormally
  - Priority checks:
  - `which_arm: right`
  - Whether `obs_key_map` corresponds to actual topics on real machines
  - Whether `pretrained_path` is the correct single right arm LingBot checkpoint
  - `go.bag` Whether to adapt to the current robot posture

- `go.bag` playback is normal, but `run` is abnormal
  - It indicates that there is no problem with the trajectory link. Focus on checking the model input, checkpoint and observation dimensions.

### 9.8 Related documents

- Deployment configuration: `configs/deploy/kuavo_env.yaml`
- Real machine script: `kuavo_deploy/src/scripts/script.py`
- Real machine reasoning entrance: `kuavo_deploy/src/eval/real_single_test.py`
- LingBot deployment adapter: `kuavo_deploy/utils/lingbot_adapter.py`

