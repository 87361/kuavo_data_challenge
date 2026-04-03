# LingBot real machine testing process

## Robot startup process

Connect to Kuavo-manipulation network, password: manipulation
You can log in to the backend interface through http://192.168.5.1/ to view the IP of the connected device.

Ensure that the upper and lower computers and side devices can communicate with each other on the same network segment

Connect to the robot lower computer: ssh lab@192.168.5.117 Password: three spaces

`cd kuavo-ros-control`

`sudo su`

`roslaunch humanoid_controllers load_kuavo_real.launch` starts the motion control node

Connect to the robot host computer: ssh leju_kuavo@192.168.5.111 Password: leju_kuavo

The local 4090 computer needs to be configured with the host computer as master and slave.

Start the camera on the host computer

`sudo systemctl start start_camera.service`

`sudo systemctl restart start_camera.service`

If you find that the camera cannot start, it may be a problem with the ROS_MASTER_URI of the host. You need to modify /etc/kuavo.conf and change the ROS_MASTER_URI from pointing to the local machine to pointing to the machine where the remote ROS Master is located. Do not modify to ROS_IP

If you start python kuavo_deploy/src/scripts/script.py on the side machine and cannot reason, and keep reporting an error that the topic is empty and waiting, but the host machine, the slave machine and the side machine can ping each other, then you need to modify the /etc/hosts of the side machine and add kuavo_master
```
127.0.0.1	localhost
127.0.1.1	myl
192.168.5.165   kuavo_master
```
If the side computer message is missing, you can directly copy kuavo-ros-control from the lower computer.

Head control:

After source from any warehouse on the lower computer (my own is kuavo-ros-control):

`rostopic pub /robot_head_motion_data kuavo_msgs/robotHeadMotionData "joint_data: [0.0, 27.0]" --once`

-Restore zero position

`python kuavo_deploy/src/scripts/script.py --task back_to_zero --config configs/deploy/kuavo_env.lmy_go_run.yaml`

- Replay bag tests

`python kuavo_deploy/src/scripts/script.py --task go --config configs/deploy/kuavo_env.lmy_go_run.yaml`

- Model inference testing
`python kuavo_deploy/src/scripts/script.py --task run --config configs/deploy/kuavo_env.lmy_go_run.yaml`

---

This document is for the real machine deployment test based on LingBot in the current warehouse. It is assumed that you have completed:

- LingBot model training;
- `configs/deploy/kuavo_env.yaml` has been changed to a single right-arm deployment configuration according to the current data set;
- The usage environment is `kdc_dev`;
- The real machine side already has the running conditions of `kuavo-ros-opensource`.

The data characteristics corresponding to the current configuration are:

- `observation.images.head_cam_h`
- `observation.images.wrist_cam_r`
- `observation.state`
- `action`

That is, the single-arm task of "head camera + right wrist camera + right arm status/action".

## 1. Check before boarding the machine

First make sure the following items are in place:

- The current environment is `kdc_dev`
- `policy_type` in `configs/deploy/kuavo_env.yaml` is `lingbot`
- `go_bag_path` in `configs/deploy/kuavo_env.yaml` has been changed to the real absolute path
- `pretrained_path` in `configs/deploy/kuavo_env.yaml` points to the real LingBot `hf_ckpt`
- The real machine ROS topic and configuration are consistent

It is recommended to check the model path first:

```bash
ls /home/yunxi/lmy/VLA/kuavo_data_challenge/outputs/train/lingbot_task/lingbot_post_train/run_20260314_061741/checkpoints/global_step_15300/hf_ckpt
```

## 2. Start the bottom layer of the real machine

Execute in deployment terminal:

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
roslaunch humanoid_controllers load_kuavo_real.launch
```

If this is the first time you turn on your computer and need calibration:

```bash
roslaunch humanoid_controllers load_kuavo_real.launch cali:=true
```

## 3. Check ROS observation link

Execute in a new terminal:

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
rostopic hz /cam_h/color/image_raw/compressed
rostopic hz /cam_r/color/image_raw/compressed
rostopic echo /sensors_data_raw
rostopic echo /leju_claw_state
```

At least confirm:

- The head RGB has data;
- Right wrist RGB has data;
- There is data on joint status;
- The gripper status has data.

If these topics are inconsistent with `configs/deploy/kuavo_env.yaml`, change the configuration before continuing.

## 4. The first stage of testing: only test `go.bag`

Don't run the model yet, just measure reaching the starting position:

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py \
  --task go \
  --config configs/deploy/kuavo_env.yaml
```

This step is explained by:

- `go_bag_path` is readable;
- The track playback link is normal;
- There are no obvious errors in the current right arm and gripper control mapping;
- The starting position is at least reachable.

If this step is unsafe, do not continue running the model.

## 5. Second stage test: run the model directly from the current position

Manually place the robot in a safe starting position, and first reason directly from the current position without moving `go.bag`:

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py \
  --task run \
  --config configs/deploy/kuavo_env.yaml
```

This step mainly verifies:

- Whether LingBot checkpoint can be loaded;
- Whether the head + right wrist + 8-dimensional state can enter the model normally;
- Can the model output actions be sent to the real machine;
- Whether the direction of action is basically reasonable.

Suggestions for first test on real machine:

- `eval_episodes=1`
- Someone is on duty nearby
- Be prepared for physical emergency stops at any time
- Only look at whether the first few steps are reasonable

## 6. The third phase of testing: complete `go_run`

After the current two steps have passed, perform the complete test:

```bash
conda activate kdc_dev
source /opt/ros/noetic/setup.bash
python kuavo_deploy/src/scripts/script.py \
  --task go_run \
  --config configs/deploy/kuavo_env.yaml
```

It will execute:

1. First press `go.bag` to get to the mission starting posture
2. Then start LingBot inference

This is the closest process to a formal deployment.

## 7. Pause, stop and return to zero

`script.py` will print the current process PID after starting.

Pause/Resume:

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
python kuavo_deploy/src/scripts/script.py \
  --task back_to_zero \
  --config configs/deploy/kuavo_env.yaml
```

It is recommended to verify `back_to_zero` separately before the first real machine test.

## 8. Recommended test order

It is recommended to strictly follow the following order:

1. Start the bottom layer of the real machine
2. Check out 4 key topics
3. Test `go`
4. Test `run`
5. Final test `go_run`

Don’t jump right into `go_run` from the beginning.

## 9. FAQ

- `Robotic arm initialization failed`
  - Prioritize checking whether the version of `kuavo-humanoid-sdk` is consistent with the robot side.

- `LingBot import failed`
  - Confirm that the current environment is `kdc_dev`, and that LingBot related modules can be imported normally in this environment.

- `wrist_cam_l is missing`
  - The current repository has modified `lingbot_adapter.py` to allow deployment when only the right wrist image is available.

- The model can be started but behaves abnormally
  - Priority checks:
    - `which_arm: right`
    - Whether `obs_key_map` corresponds to actual topics on real machines
    - Whether `pretrained_path` is the correct single right arm LingBot checkpoint
    - `go.bag` Whether to adapt to the current robot posture

- `go.bag` playback is normal, but `run` is abnormal
  - It indicates that there is no problem with the trajectory link. Focus on checking the model input, checkpoint and observation dimensions.

## 10. Key documents

- Deployment configuration: [configs/deploy/kuavo_env.yaml](/home/yunxi/lmy/VLA/kuavo_data_challenge/configs/deploy/kuavo_env.yaml)
- Real device script: [kuavo_deploy/src/scripts/script.py](/home/yunxi/lmy/VLA/kuavo_data_challenge/kuavo_deploy/src/scripts/script.py)
- Real machine reasoning entrance: [kuavo_deploy/src/eval/real_single_test.py](/home/yunxi/lmy/VLA/kuavo_data_challenge/kuavo_deploy/src/eval/real_single_test.py)
- LingBot deployment adapter: [kuavo_deploy/utils/lingbot_adapter.py](/home/yunxi/lmy/VLA/kuavo_data_challenge/kuavo_deploy/utils/lingbot_adapter.py)
