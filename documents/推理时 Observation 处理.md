推理时 Observation 处理分析报告
1. arm_state 的处理逻辑
位置: /home/yly/ICRA-kuavo/kuavo_data_challenge/kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py
处理流程:
在 get_obs() (452-504行) 中处理：
从 obs_buffer 获取原始观测数据
根据 arm_state_keys 配置（默认 ["joint_q", "gripper"]）提取状态
根据 which_arm 参数（'both'/'left'/'right'）分割左右臂数据：
both: 数据按中点分割，左臂取前半部分，右臂取后半部分
left: 只取左臂数据
right: 只取右臂数据
拼接为 observation.state，格式：[左臂joint_q(7维) + 左臂gripper(1维) + 右臂joint_q(7维) + 右臂gripper(1维)]
关键代码:
KuavoBaseRosEnv.py
Lines 478-500
state_keys = [k for k in self.arm_state_keys if k in self.arm_state]arm_data = { "left": [], "right": [] }for key in state_keys:    data = self.arm_state[key]    if len(data) == 0:        continue    mid = len(data) // 2    if self.which_arm == "both":        arm_data["left"].append(data[:mid])        arm_data["right"].append(data[mid:])    elif self.which_arm == "left":        arm_data["left"].append(data)    elif self.which_arm == "right":        arm_data["right"].append(data)# 拼接结果obs["observation.state"] = np.concatenate(    arm_data["left"] + arm_data["right"], axis=0)
配置位置: configs/deploy/kuavo_env.yaml:42
arm_state_keys: ["joint_q","gripper"]  # Do not change. Model observation arm state
2. 图像格式（RGB/BGR）处理
位置: /home/yly/ICRA-kuavo/kuavo_data_challenge/kuavo_deploy/utils/obs_buffer.py
处理流程:
在 rgb_callback() (217-238行) 中处理：
使用 cv2.imdecode() 解码压缩图像（OpenCV 默认 BGR）
使用 cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) 转换为 RGB
可选的 resize（根据配置）
使用 to_tensor() 转换为 PyTorch tensor
关键代码:
obs_buffer.py
Lines 217-238
def rgb_callback(self, msg: CompressedImage, key: str, handle: dict):    # 解码    img_arr = np.frombuffer(msg.data, dtype=np.uint8)    cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)        # 颜色转换和resize    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB    resize_wh = handle.get("params", {}).get("resize_wh", None)    if resize_wh:        cv_img = cv2.resize(cv_img, resize_wh)    data = self.img_preprocess(cv_img)  # 转换为tensor
数据流:
ROS CompressedImage (BGR) → OpenCV解码 (BGR) → BGR2RGB转换 → PyTorch Tensor (RGB, C×H×W)
3. Task Description 的处理
位置: /home/yly/ICRA-kuavo/kuavo_data_challenge/kuavo_deploy/src/eval/sim_auto_test.py
处理流程:
在 create_task_injecting_preprocessor() (166-189行) 中处理：
仅对 PI05/PI0 策略注入 task_description
从配置的 task_description 获取，默认使用 task 名称
在 preprocessor 中将 task 字段注入到 observation
关键代码:
sim_auto_test.py
Lines 166-189
def create_task_injecting_preprocessor(preprocessor, task_description, policy_type):    if policy_type in ["pi05", "pi0"]:  # 需要 task 字段的策略        def wrapped_preprocessor(observation):            batch_input = {**observation, "task": [task_description]}            return preprocessor(batch_input)        return wrapped_preprocessor    else:        return preprocessor  # 其他策略直接返回原始 preprocessor
配置位置: configs/deploy/kuavo_env.yaml:96
task_description: "Pick and Place"  # Task description for PI05 model
使用位置: sim_auto_test.py:198
task_description = getattr(cfg, 'task_description', None) or task
4. 关节顺序定义
关节名称定义:
位置: /home/yly/ICRA-kuavo/kuavo_data_challenge/kuavo_data/common/kuavo_dataset.py:19-22
DEFAULT_ARM_JOINT_NAMES = [    "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link",     "zarm_l5_link", "zarm_l6_link", "zarm_l7_link",  # 左臂7个关节    "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link",     "zarm_r5_link", "zarm_r6_link", "zarm_r7_link",  # 右臂7个关节]
关节索引提取:
位置: /home/yly/ICRA-kuavo/kuavo_data_challenge/kuavo_deploy/utils/obs_buffer.py:278
从完整的 28 维关节数据中提取双臂的 14 个关节（索引 12-26）：
obs_buffer.py
Lines 268-287
def sensorsData_callback(self, msg: sensorsData, key: str, handle = dict):    joint = msg.joint_data.joint_q    timestamp = msg.header.stamp.to_sec()    # FK 计算需要双臂的14个关节（索引12-26）    arm_joints = joint[12:26]  # 提取双臂关节    self.compute_dependent_obs(key, arm_joints, timestamp)
关节顺序说明:
完整关节数组（28维）: [leg_joints(12) + arm_joints(14) + head_joints(2)]
双臂关节（14维，索引12-26）: [左臂7个 + 右臂7个]
在 get_obs() 中，根据 which_arm 进一步分割：
both: [左臂7个joint + 左臂1个gripper + 右臂7个joint + 右臂1个gripper] = 16维
left: [左臂7个joint + 左臂1个gripper] = 8维
right: [右臂7个joint + 右臂1个gripper] = 8维
相关文件:
/home/yly/ICRA-kuavo/kuavo_data_challenge/kuavo_deploy/utils/obs_buffer.py - 观测缓冲处理
/home/yly/ICRA-kuavo/kuavo_data_challenge/kuavo_deploy/src/eval/sim_auto_test.py - 推理测试脚本
/home/yly/ICRA-kuavo/kuavo_data_challenge/kuavo_deploy/kuavo_env/KuavoBaseRosEnv.py - 环境基类
/home/yly/ICRA-kuavo/kuavo_data_challenge/configs/deploy/kuavo_env.yaml - 配置文件
以上为推理时 Observation 处理的分析结果。