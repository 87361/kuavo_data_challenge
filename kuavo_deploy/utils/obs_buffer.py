from dataclasses import asdict
from collections import deque
from typing import Dict, Callable, Any, Optional, List
import torch
import numpy as np
import cv2
import rospy
from tqdm import tqdm
import time
import sys
import statistics
from kuavo_deploy.config import KuavoConfig
from sensor_msgs.msg import CompressedImage, JointState
from torchvision.transforms.functional import to_tensor
from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import sensorsData,lejuClawState
from kuavo_deploy.utils.signal_controller import ControlSignalManager
from kuavo_deploy.utils.logging_utils import setup_logger
from kuavo_deploy.utils.ros_manager import ROSManager


log_robot = setup_logger("robot")


class TimingCollector:
    """可选的延迟收集器，用于诊断性能问题"""
    
    def __init__(self, enabled: bool = False, max_samples: int = 1000):
        self.enabled = enabled
        self.max_samples = max_samples
        self._timings: Dict[str, deque] = {}
    
    def record(self, name: str, time_ms: float):
        """记录一个计时样本"""
        if not self.enabled:
            return
        if name not in self._timings:
            self._timings[name] = deque(maxlen=self.max_samples)
        self._timings[name].append(time_ms)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取指定名称的统计数据"""
        if name not in self._timings or len(self._timings[name]) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
        times = list(self._timings[name])
        return {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "count": len(times),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有计时的统计数据"""
        return {name: self.get_stats(name) for name in self._timings}
    
    def reset(self):
        """重置所有计时数据"""
        self._timings.clear()
    
    def print_report(self):
        """打印计时报告"""
        if not self.enabled or not self._timings:
            print("计时未启用或无数据")
            return
        
        print("\n" + "=" * 60)
        print("           ObsBuffer 延迟诊断报告")
        print("=" * 60)
        
        for name in sorted(self._timings.keys()):
            stats = self.get_stats(name)
            print(f"  {name}:")
            print(f"    平均: {stats['mean']:.2f} ms, 标准差: {stats['std']:.2f} ms")
            print(f"    最小: {stats['min']:.2f} ms, 最大: {stats['max']:.2f} ms")
            print(f"    样本数: {stats['count']}")
        
        print("=" * 60)


class ObsBuffer:
    def __init__(
        self, 
        config: KuavoConfig,  
        obs_key_map: Dict[str, Dict[str, Any]] = None,
        compute_func_map: Dict[str, Callable] = None,
        enable_timing: bool = False,  # 新增：启用延迟测量
    ) -> None:
        self.control_signal_manager = ControlSignalManager()
        self.ros_manager = ROSManager()
        
        # === 延迟诊断收集器 Timing Collector for Diagnostics ===
        self.timing = TimingCollector(enabled=enable_timing)
        
        # === 从 KuavoConfig 中提取环境配置 Extract Configuration===
        # #region agent log
        import json as _json; open('/home/yly/ICRA-kuavo/kuavo_data_challenge/.cursor/debug.log','a').write(_json.dumps({"hypothesisId":"A","location":"obs_buffer.py:98","message":"config type check","data":{"config_type":str(type(config)),"has_env":hasattr(config,'env'),"is_dict":isinstance(config,dict)},"timestamp":time.time(),"sessionId":"debug-session"})+'\n')
        # #endregion
        # 兼容两种配置格式: KuavoConfig (有.env属性) 或直接的env配置对象
        if hasattr(config, 'env'):
            env_cfg = config.env
        else:
            env_cfg = config
        # #region agent log
        import json as _json; open('/home/yly/ICRA-kuavo/kuavo_data_challenge/.cursor/debug.log','a').write(_json.dumps({"hypothesisId":"B","location":"obs_buffer.py:106","message":"env_cfg type check","data":{"env_cfg_type":str(type(env_cfg)),"is_dict":isinstance(env_cfg,dict)},"timestamp":time.time(),"sessionId":"debug-session"})+'\n')
        # #endregion
        # 兼容字典和对象两种访问方式的辅助函数
        def get_cfg_val(cfg, key, default=None):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)
        
        self.which_arm = get_cfg_val(env_cfg, 'which_arm', 'both')

        # === 观测定义 Observation Defn ===
        self.obs_key_map = obs_key_map or get_cfg_val(env_cfg, 'obs_key_map') or {}
        self.compute_func_map = compute_func_map or {}
        
        # === 兼容旧的列表格式，转换为字典格式 Normalize obs_key_map format ===
        self.obs_key_map = self._normalize_obs_key_map(self.obs_key_map)

        # === 区分订阅型与计算型观测 Differentiate Observation Types===
        self.subscribe_keys = {k: v for k, v in self.obs_key_map.items() if v.get("type") != "computed"}
        self.computed_keys  = {k: v for k, v in self.obs_key_map.items() if v.get("type") == "computed"}

        # === 反向依赖索引 Reverse Indicing Dependencies ===
        self.source_to_computed = {}
        for comp_key, comp_info in self.computed_keys.items():
            src = comp_info.get("source")
            if src:
                self.source_to_computed.setdefault(src, []).append(comp_key)
                log_robot.info(f"Registered computed obs '{comp_key}' depends on '{src}'")

        # === 初始化观测缓存 Init Observation Buffer ===
        self.obs_buffer_size = {k: v["frequency"] for k, v in self.obs_key_map.items()}
        self.obs_buffer_data = {
            k: {"data": deque(maxlen=v["frequency"]), "timestamp": deque(maxlen=v["frequency"])}
            for k, v in self.obs_key_map.items()
        }

        # === ROS topic 对应表 Reference List ===
        self.callback_key_map = {
            '/cam_h/color/image_raw/compressed': self.rgb_callback,
            '/cam_l/color/image_raw/compressed': self.rgb_callback,
            '/cam_r/color/image_raw/compressed': self.rgb_callback,
            '/cam_h/depth/image_raw/compressedDepth': self.depth_callback,
            '/cam_l/depth/image_rect_raw/compressedDepth': self.depth_callback,
            '/cam_r/depth/image_rect_raw/compressedDepth': self.depth_callback,
            '/sensors_data_raw': self.sensorsData_callback,
            '/dexhand/state': self.qiangnaoState_callback,
            '/leju_claw_state': self.lejuClawState_callback,
            '/gripper/state': self.rq2f85State_callback,
        }
        self.setup_subscribers()

    def _normalize_obs_key_map(self, obs_key_map: dict) -> dict:
        """将旧的列表格式 obs_key_map 转换为新的字典格式
        
        旧格式 (列表): [topic, msg_type, frequency, width, height, depth_min, depth_max]
        新格式 (字典): {"topic": ..., "msg_type": ..., "frequency": ..., "image_size": [...], "depth_range": [...]}
        """
        normalized = {}
        for key, value in obs_key_map.items():
            if isinstance(value, list):
                # 列表格式: [topic, msg_type, frequency, width, height, depth_min, depth_max]
                entry = {
                    "topic": value[0],
                    "msg_type": value[1],
                    "frequency": value[2],
                }
                if len(value) >= 5:
                    entry["image_size"] = [value[3], value[4]]
                if len(value) >= 7:
                    entry["depth_range"] = [value[5], value[6]]
                normalized[key] = entry
            elif isinstance(value, dict):
                # 已经是字典格式，直接使用
                normalized[key] = value
            else:
                raise ValueError(f"Unsupported obs_key_map format for '{key}': {type(value)}")
        return normalized

    # ===== ROS订阅 Subscription =====
    def create_callback(self, callback, topic_key, handle):
        return lambda msg: callback(msg, topic_key, handle)

    def setup_subscribers(self):
        """仅订阅来自 ROS 的观测"""
        msg_type_dict = {"CompressedImage":CompressedImage,
                         "sensorsData":sensorsData,
                         "JointState":JointState,
                         "lejuClawState":lejuClawState}
        for topic_key, info in self.subscribe_keys.items():
            topic_name = info["topic"]
            assert info["msg_type"] in msg_type_dict, f"msg_type '{info['msg_type']}' is not supported; valid keys: {list(msg_type_dict.keys())}"
            msg_type = msg_type_dict[info["msg_type"]]
            callback = self.callback_key_map.get(topic_name)

            if not msg_type or not callback:
                log_robot.warning(f"Missing msg_type or callback for {topic_name}")
                continue

            handle = info.get("handle", {})
            self.ros_manager.register_subscriber(
                topic_name, msg_type, self.create_callback(callback, topic_key, handle)
            )
            log_robot.info(f"Subscribed to {topic_name} for key '{topic_key}'")

    # ===== 数据预处理 Data Preprocessing =====
    def img_preprocess(self, image):
        """图像预处理"""
        return to_tensor(image).unsqueeze(0)

    def depth_preprocess(self, depth, depth_range=[0, 1500]):
        """深度图像预处理"""
        depth_uint16 = torch.tensor(depth, dtype=torch.float32).clamp(*depth_range).unsqueeze(0)
        max_depth = depth_uint16.max()
        min_depth = depth_uint16.min()
        depth_normalized = (depth_uint16 - min_depth) / (max_depth - min_depth + 1e-9)
        return depth_normalized

    # ===== Callback 函数群 Functions =====
    def rgb_callback(self, msg: CompressedImage, key: str, handle: dict):
        callback_start = time.perf_counter()
        
        # 解码
        decode_start = time.perf_counter()
        img_arr = np.frombuffer(msg.data, dtype=np.uint8)
        cv_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise ValueError("Failed to decode compressed image")
        self.timing.record("rgb_decode", (time.perf_counter() - decode_start) * 1000)
        
        # 颜色转换和resize
        preprocess_start = time.perf_counter()
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        resize_wh = handle.get("params", {}).get("resize_wh", None)
        if resize_wh:
            cv_img = cv2.resize(cv_img, resize_wh)
        data = self.img_preprocess(cv_img)
        self.timing.record("rgb_preprocess", (time.perf_counter() - preprocess_start) * 1000)
        
        self._append_data(key, data, msg.header.stamp.to_sec())
        self.timing.record("rgb_callback_total", (time.perf_counter() - callback_start) * 1000)

    def depth_callback(self, msg: CompressedImage, key: str, handle: dict):
        callback_start = time.perf_counter()
        
        png_magic = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        idx = msg.data.find(png_magic)
        if idx == -1:
            raise ValueError("Invalid depth message, PNG header not found")
        
        # 解码
        decode_start = time.perf_counter()
        np_arr = np.frombuffer(msg.data[idx:], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if image is None:
            return
        self.timing.record("depth_decode", (time.perf_counter() - decode_start) * 1000)
        
        # 预处理
        preprocess_start = time.perf_counter()
        resize_wh = handle.get("params", {}).get("resize_wh", None)
        if resize_wh:
            image = cv2.resize(image, resize_wh)
        image = image[np.newaxis, ...]
        data = self.depth_preprocess(image, depth_range=handle.get("params", {}).get("depth_range", [0, 1500]))
        self.timing.record("depth_preprocess", (time.perf_counter() - preprocess_start) * 1000)
        
        self._append_data(key, data, msg.header.stamp.to_sec())
        self.timing.record("depth_callback_total", (time.perf_counter() - callback_start) * 1000)

    def sensorsData_callback(self, msg: sensorsData, key: str, handle = dict):
        callback_start = time.perf_counter()
        
        # Float64Array ()
        joint = msg.joint_data.joint_q
        timestamp = msg.header.stamp.to_sec()

        # FK 计算需要双臂的14个关节（索引12-26）
        # 计算依赖于此数据源的观测（例如 eef_pose）
        fk_start = time.perf_counter()
        arm_joints = joint[12:26]  # 提取双臂关节
        self.compute_dependent_obs(key, arm_joints, timestamp)
        self.timing.record("sensorsData_fk_compute", (time.perf_counter() - fk_start) * 1000)

        slice_value = handle.get("params", {}).get("slice", None)
        if slice_value is not None:
            joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, timestamp)
        self.timing.record("sensorsData_callback_total", (time.perf_counter() - callback_start) * 1000)

    def lejuClawState_callback(self, msg: lejuClawState, key: str, handle = dict):
        # Float64Array ()
        joint = msg.data.position
        slice_value = handle.get("params", {}).get("slice", None)
        if slice_value is not None:
            joint = [x / 100 for slc in slice_value for x in joint[slc[0]:slc[1]]] # 注意缩放
        else:
            joint = [x / 100 for x in joint]  # 无切片时直接缩放
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, msg.header.stamp.to_sec())

    def qiangnaoState_callback(self, msg: JointState, key: str, handle = dict):
        joint = msg.position
        joint = [figure / 100 for figure in joint]
        slice_value = handle.get("params", {}).get("slice", None)
        if slice_value is not None:
            joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, msg.header.stamp.to_sec())

    def rq2f85State_callback(self, msg: JointState, key: str, handle = dict):
        joint = msg.position
        joint = [figure / 0.8 for figure in joint]
        slice_value = handle.get("params", {}).get("slice", None)
        if slice_value is not None:
            joint = [x for slc in slice_value for x in joint[slc[0]:slc[1]]]
        # joint = torch.tensor(joint, dtype=torch.float32, device=self.device)
        self._append_data(key, joint, msg.header.stamp.to_sec())

    # ===== 公共方法 Public Methods =====
    def _append_data(self, key, data, timestamp):
        self.obs_buffer_data[key]["data"].append(data)
        self.obs_buffer_data[key]["timestamp"].append(timestamp)

    def compute_dependent_obs(self, source_key, source_data, timestamp):
        for comp_key in self.source_to_computed.get(source_key, []):
            func = self.compute_func_map.get(comp_key)
            if not func:
                log_robot.warning(f"No compute function for {comp_key}")
                continue
            try:
                compute_start = time.perf_counter()
                data = func(source_data, which_arm=self.which_arm)
                self.timing.record(f"compute_{comp_key}", (time.perf_counter() - compute_start) * 1000)
                if data is not None:
                    self._append_data(comp_key, data, timestamp)
            except Exception as e:
                log_robot.error(f"Error computing {comp_key} from {source_key}: {e}")

    def obs_buffer_is_ready(self):
        return all(len(self.obs_buffer_data[k]["data"]) == self.obs_key_map[k]["frequency"] for k in self.obs_key_map)

    def stop_subscribers(self):
        self.ros_manager.close()

    def wait_buffer_ready(self):
        progress = {k: 0 for k in self.obs_key_map}
        total = {k: v["frequency"] for k, v in self.obs_key_map.items()}
        last_log_time = 0

        while not self.obs_buffer_is_ready():
            if not self.control_signal_manager.check_control_signals():
                log_robot.info("🛑 Stop signal detected, exiting")
                sys.exit(1)

            now = time.time()
            # 每隔 1 秒打印一次日志
            if now - last_log_time > 0.2:
                logs = []
                for k in progress:
                    new_len = len(self.obs_buffer_data[k]["data"])
                    progress[k] = new_len
                    logs.append(f"{k}: {new_len}/{total[k]}")
                log_robot.info(" | ".join(logs))
                last_log_time = now

            time.sleep(0.1)

        log_robot.info("✅ All buffers ready!")
        return True

    def get_latest_obs(self):
        start = time.perf_counter()
        obs = {}
        for k, buf in self.obs_buffer_data.items():
            obs[k] = list(buf["data"])[-1]  # 取最新一帧
        self.timing.record("get_latest_obs", (time.perf_counter() - start) * 1000)
        return obs

    def get_aligned_obs(self, reference_keys=["/cam_h/color/image_raw/compressed"], max_dt=0.01, ratio=1.0):
        """
        返回各观测时间上对齐的最新帧
        reference_keys: 以哪些key作为时间参考，默认 None -> 所有 key 最小的最新时间戳
        max_dt: 最大允许时间偏差（秒），超出则返回 None
        """
        total_start = time.perf_counter()
        
        # ===== 获取参考时间戳 =====
        ref_start = time.perf_counter()
        if reference_keys:
            # 用指定 key 的最新时间戳
            ref_times = []
            for k in reference_keys:
                buf = self.obs_buffer_data[k]
                if len(buf) == 0:
                    continue
                _, ts = buf["data"], buf["timestamp"]
                ref_times.append(ts[-1])
            if not ref_times:
                return None
            ref_time = min(ref_times)  # 也可以取 min 或 max，根据需要
        else:
            # 没有指定 key -> 所有观测的最新 timestamp 的最小值
            last_timestamps = []
            for buf in self.obs_buffer_data.values():
                if len(buf) == 0:
                    continue
                _, ts = buf["data"], buf["timestamp"]
                last_timestamps.append(ts[-1])
            if not last_timestamps:
                return None
            ref_time = np.min(last_timestamps)
        self.timing.record("aligned_get_ref_time", (time.perf_counter() - ref_start) * 1000)

        # ===== 对齐各观测 =====
        align_start = time.perf_counter()
        aligned_obs = {}
        for k, buf in self.obs_buffer_data.items():
            n = int(len(buf["data"]) * ratio)
            data, ts = list(buf["data"])[-n:], list(buf["timestamp"])[-n:]
            ts = np.array(ts)
            if len(ts) == 0:
                aligned_obs[k] = None
                continue
            idx = np.argmin(np.abs(ts - ref_time))
            if abs(ts[idx] - ref_time) > max_dt:
                aligned_obs[k] = None
            else:
                aligned_obs[k] = data[idx]
        self.timing.record("aligned_search_obs", (time.perf_counter() - align_start) * 1000)

        self.timing.record("get_aligned_obs_total", (time.perf_counter() - total_start) * 1000)
        return aligned_obs

