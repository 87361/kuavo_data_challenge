"""
Chunked streaming rosbag module

Core idea (refer to the on-demand reading method of Diffusion Policy):
1. The first scan: only reads the timestamp and determines the main timeline (no image data is loaded, and the memory usage is very small)
2. The second scan: read in blocks according to the time window, and write the dataset while aligning while reading.

Differences from the original method:
- Raw: Load all data into memory at once→ Alignment→ Writing to dataset (memory peak is huge)
- New method: chunked reading→ Instant alignment→ instant write→ Release memory (memory usage is controllable)
"""

import numpy as np
import rosbag
from collections import defaultdict
from typing import Dict, List, Callable, Optional, Tuple, Generator
import logging
import bisect

logger = logging.getLogger(__name__)


class ChunkedRosbagProcessor:
    """
    Blocked streaming processing of rosbag enables reading, alignment and processing at the same time
    
    Workflow:
    1. scan_timestamps(): The first scan only reads the timestamp (minimum memory usage)
    2. process_chunks(): The second scan is performed in blocks according to time windows.
    """
    
    def __init__(self, msg_processer, topic_process_map: dict, 
                 camera_names: list, train_hz: int, main_timeline: str, main_timeline_fps: int, 
                 sample_drop: int, only_half_up_body: bool):
        self._msg_processer = msg_processer
        self._topic_process_map = topic_process_map
        self.camera_names = camera_names
        self.train_hz = train_hz
        self.main_timeline_fps = main_timeline_fps
        self.sample_drop = sample_drop
        self.only_half_up_body = only_half_up_body
        self.main_timeline = main_timeline
    
    def scan_timestamps_only(self, bag_file: str) -> Tuple[str, List[float], Dict[str, List[float]]]:
        """
        First pass of scan: only reads timestamps, does not load data
        
        Memory footprint: only timestamp list (a few MB), no image data
        
        Returns:
            main_timeline: Main timeline topic key
            main_timestamps: Aligned primary timestamp sequence (after downsampling)
            all_timestamps: List of original timestamps for each topic
        """
        bag = self._load_bag(bag_file)
        
        #Count the timestamp of each topic (without loading message content)
        all_timestamps = defaultdict(list)
        topic_to_key = {}
        for k, v in self._topic_process_map.items():
            if v["topic"] not in topic_to_key.keys():
                topic_to_key[v["topic"]] = [k]
            else:
                topic_to_key[v["topic"]].append(k)
        
        logger.info(f"[Phase 1] Scanning timestamps from {bag_file}...")
        
        #Traverse once and collect the timestamps of all topics
        all_topics = [v["topic"] for v in self._topic_process_map.values()]
        for topic, msg, t in bag.read_messages(topics=all_topics):
            keys = topic_to_key.get(topic)
            for key in keys:
                all_timestamps[key].append(t.to_sec())
                # all_timestamps[key].append(msg.header.stamp.to_sec())
                # all_timestamps[key].append(msg.header.stamp.to_sec() if hasattr(msg, 'header') and hasattr(msg.header, 'stamp') else t.to_sec())
        
        bag.close()                           
        
        #Determine the main timeline: the camera with the most messages
        camera_counts = {k: len(all_timestamps.get(k, [])) for k in self.camera_names}
        if not any(camera_counts.values()):
            raise ValueError("No camera data found in rosbag")
        if self.main_timeline is None:
            main_timeline = max(camera_counts, key=lambda k: camera_counts[k])
        else:
            main_timeline = self.main_timeline
        logger.info(f"Main timeline: {main_timeline} ({camera_counts[main_timeline]} frames)")
        
        #Generate an aligned primary timestamp sequence
        jump = self.main_timeline_fps // self.train_hz
        raw_timestamps = all_timestamps[main_timeline]

        if len(raw_timestamps) < 2 * self.sample_drop + 1:
            raise ValueError(f"Not enough frames: {len(raw_timestamps)}")

        #Discard the first and last frames and downsample
        #Note: When sample_drop is 0, [self.sample_drop:-self.sample_drop] cannot be used,
        #Because [-0] is equivalent to [0], the slice result will be empty.
        if self.sample_drop > 0:
            main_timestamps = raw_timestamps[self.sample_drop:-self.sample_drop][::jump]
        else:
            main_timestamps = raw_timestamps[::jump]
        
        #Get the "earliest end" time among all topics
        min_end = min(
            ts_list[-1]
            for ts_list in all_timestamps.values()
            if len(ts_list) > 0
        )

        #Cut the main timeline to only keep the time point when all topics still have data
        before_len = len(main_timestamps)
        main_timestamps = [t for t in main_timestamps if t < min_end]
        after_len = len(main_timestamps)

        logger.info(
            f"Trim main timeline by min_end={min_end:.6f}, "
            f"frames: {before_len} -> {after_len}"
        )
        
        logger.info(f"Generated {len(main_timestamps)} aligned timestamps "
                   f"(from {len(raw_timestamps)} raw frames, "
                   f"dropped {self.sample_drop} frames at each end, jump={jump})")

        return main_timeline, main_timestamps, dict(all_timestamps)
    
    def process_in_chunks(
        self,
        bag_file: str,
        main_timestamps: List[float],
        all_timestamps: Dict[str, List[float]],
        frame_callback: Callable[[dict, int], None],
        chunk_size: int = 100,
        save_callback: Optional[Callable[[], None]] = None
    ) -> int:
        """
        Second pass of scanning: processing in blocks according to time window
        
        Strategy:
        1. Divide main_timestamps into multiple chunks
        2. For each chunk, only messages within that time range are read
        3. frame_callback is called immediately after alignment
        4. After each chunk is processed, save_callback is called to release the memory.
        
        Args:
            bag_file: rosbagfile path
            main_timestamps: Aligned primary timestamp sequence
            all_timestamps: List of original timestamps for each topic (for quick lookups)
            frame_callback: Callback function to handle each frame (aligned_frame, frame_idx) -> None
            chunk_size: The number of frames contained in each chunk
            save_callback: Callback after each chunk is processed (used to save and release memory)
        
        Returns:
            Total frames processed
        """
        bag = self._load_bag(bag_file)
        
        #Build a timestamp index for each topic (for quickly finding recent frames)
        timestamp_arrays = {k: np.array(v) for k, v in all_timestamps.items()}
        
        #Precompute each topic index corresponding to each primary timestamp (to avoid repeated searches)
        alignment_indices = self._precompute_alignment_indices(
            main_timestamps, timestamp_arrays
        )
        
        #Detect timestamp gaps in kuavo_arm_traj
        arm_traj_gaps = self._detect_arm_traj_gaps(all_timestamps)
        
        num_chunks = (len(main_timestamps) + chunk_size - 1) // chunk_size
        total_frames = 0
        
        logger.info(f"[Phase 2] Processing {len(main_timestamps)} frames in {num_chunks} chunks...")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(main_timestamps))
            chunk_timestamps = main_timestamps[start_idx:end_idx]
            
            if not chunk_timestamps:
                continue
            
            #Determine the time range for that chunk (extend a bit to ensure aligned data is available)
            time_margin = 1.0 / self.train_hz  #one frame time
            chunk_start_time = chunk_timestamps[0] - time_margin
            chunk_end_time = chunk_timestamps[-1] + time_margin
            
            logger.debug(f"Chunk {chunk_idx+1}/{num_chunks}: "
                        f"frames {start_idx}-{end_idx-1}, "
                        f"time range [{chunk_start_time:.3f}, {chunk_end_time:.3f}]")
            
            #Read messages within this time range
            chunk_data = self._read_chunk_data(bag, chunk_start_time, chunk_end_time)
            
            #Align and process each frame
            for local_idx, (global_idx, main_stamp) in enumerate(
                zip(range(start_idx, end_idx), chunk_timestamps)
            ): 
                aligned_frame = self._align_single_frame(
                    main_stamp=main_stamp,
                    global_idx=global_idx,
                    chunk_data=chunk_data,
                    timestamp_arrays=timestamp_arrays,
                    alignment_indices=alignment_indices,
                    arm_traj_gaps=arm_traj_gaps
                )
                
                frame_callback(aligned_frame, global_idx)
                total_frames += 1
            
            #Release chunk data
            del chunk_data
            
            #Call save callback
            if save_callback:
                save_callback()
                logger.info(f"Chunk {chunk_idx+1}/{num_chunks} processed and saved. "
                           f"Frames: {start_idx}-{end_idx-1}")
        
        bag.close()
        logger.info(f"Total frames processed: {total_frames}")
        return total_frames
    
    def _precompute_alignment_indices(
        self, 
        main_timestamps: List[float], 
        timestamp_arrays: Dict[str, np.ndarray]
    ) -> Dict[str, List[int]]:
        """
        Precompute each topic index corresponding to each primary timestamp
        Use binary search, which is much faster than searching every frame
        """
        alignment_indices = {}
        
        for key, ts_array in timestamp_arrays.items():
            if len(ts_array) == 0:
                alignment_indices[key] = []
                continue
            
            indices = []
            for stamp in main_timestamps:
                #Binary search for the most recent timestamp
                idx = bisect.bisect_left(ts_array, stamp)
                if idx == 0:
                    closest_idx = 0
                elif idx == len(ts_array):
                    closest_idx = len(ts_array) - 1
                else:
                    #Choose the closer
                    if abs(ts_array[idx] - stamp) < abs(ts_array[idx-1] - stamp):
                        closest_idx = idx
                    else:
                        closest_idx = idx - 1
                indices.append(closest_idx)
            
            alignment_indices[key] = indices
        
        return alignment_indices
    
    def _detect_arm_traj_gaps(self, all_timestamps: Dict[str, List[float]]) -> List[Tuple[float, float]]:
        """Detect timestamp gaps in kuavo_arm_traj"""
        gaps = []
        if not self.only_half_up_body and "action.kuavo_arm_traj" in all_timestamps and len(all_timestamps["action.kuavo_arm_traj"]) > 0:
            timestamps = all_timestamps["action.kuavo_arm_traj"]
            if len(timestamps) > 1:
                gap_threshold = 0.15 * 10 / self.train_hz
                for i in range(1, len(timestamps)):
                    if timestamps[i] - timestamps[i-1] > gap_threshold:
                        gaps.append((timestamps[i-1], timestamps[i]))
        if len(gaps) > 0:
            logger.info(f"Detected {len(gaps)} gaps in action.kuavo_arm_traj")
        return gaps
    
    def _read_chunk_data(self, bag: rosbag.Bag, start_time: float, end_time: float) -> Dict[str, Dict[float, dict]]:
        """
        Read message data within a specified time range
        
        Returns:
            {topic_key: {timestamp: msg_data}}
        """
        import rospy
        chunk_data = defaultdict(dict)

        topic_to_key = {}
        for k, v in self._topic_process_map.items():
            if v["topic"] not in topic_to_key.keys():
                topic_to_key[v["topic"]] = [k]
            else:
                topic_to_key[v["topic"]].append(k)
        
        #Use time range filtering
        try:
            start_ros_time = rospy.Time.from_sec(start_time)
            end_ros_time = rospy.Time.from_sec(end_time)
            
            for topic, msg, t in bag.read_messages(
                topics=list(topic_to_key.keys()),
                start_time=start_ros_time,
                end_time=end_ros_time
            ):
                keys = topic_to_key.get(topic)
                for key in keys:
                    msg_process_fn = self._topic_process_map[key]["msg_process_fn"]
                    msg_data = msg_process_fn(msg)
                    msg_data["timestamp"] = t.to_sec()
                    chunk_data[key][t.to_sec()] = msg_data
                    # chunk_data[key][msg_data["timestamp"]] = msg_data

        except Exception as e:
            logger.warning(f"Time-range filtering failed: {e}, falling back to full scan")
            #Fall back to full scan + manual filtering
            for topic, msg, t in bag.read_messages(topics=list(topic_to_key.keys())):
                ts = t.to_sec()
                # ts = msg.header.stamp.to_sec()
                # ts = msg.header.stamp.to_sec() if hasattr(msg, 'header') and hasattr(msg.header, 'stamp') else t.to_sec()

                if start_time <= ts <= end_time:
                    key = topic_to_key.get(topic)
                    if key:
                        msg_process_fn = self._topic_process_map[key]["msg_process_fn"]
                        msg_data = msg_process_fn(msg)
                        msg_data["timestamp"] = ts
                        chunk_data[key][ts] = msg_data
        return dict(chunk_data)
    
    def _align_single_frame(
        self,
        main_stamp: float,
        global_idx: int,
        chunk_data: Dict[str, Dict[float, dict]],
        timestamp_arrays: Dict[str, np.ndarray],
        alignment_indices: Dict[str, List[int]],
        arm_traj_gaps: List[Tuple[float, float]]
    ) -> dict:
        """
        Align single frame data
        """
        aligned_frame = {"timestamp": main_stamp}
        
        for key in self._topic_process_map.keys():
            #Special handling of gaps in kuavo_arm_traj
            if key == "action.kuavo_arm_traj" and arm_traj_gaps:
                in_gap = any(gap_start < main_stamp < gap_end 
                            for gap_start, gap_end in arm_traj_gaps)
                if in_gap:
                    #In the gaps, fill them with 999
                    sample_data = next(iter(chunk_data.get(key, {}).values()), None)
                    if sample_data and "data" in sample_data:
                        data_dim = len(sample_data["data"])
                        aligned_frame[key] = {
                            "data": np.full(data_dim, 999.0, dtype=np.float32),
                            "timestamp": main_stamp
                        }
                    continue
            
            #Get precomputed index
            if key not in alignment_indices or global_idx >= len(alignment_indices[key]):
                aligned_frame[key] = None
                continue
            
            closest_idx = alignment_indices[key][global_idx]
            
            #Get the corresponding timestamp from timestamp_arrays
            if key not in timestamp_arrays or len(timestamp_arrays[key]) == 0:
                aligned_frame[key] = None
                continue
            
            target_ts = timestamp_arrays[key][closest_idx]
            
            #Find data from chunk_data
            if key in chunk_data:
                #Find the data closest to target_ts
                ts_list = list(chunk_data[key].keys())
                if ts_list:
                    closest_chunk_ts = min(ts_list, key=lambda x: abs(x - target_ts))
                    aligned_frame[key] = chunk_data[key][closest_chunk_ts]
                else:
                    
                    aligned_frame[key] = None
            else:
                aligned_frame[key] = None
        return aligned_frame
    
    def _load_bag(self, bag_file: str) -> rosbag.Bag:
        """Load rosbag file"""
        try:
            return rosbag.Bag(bag_file)
        except rosbag.bag.ROSBagUnindexedException:
            logger.warning(f"Bag file {bag_file} is unindexed, attempting to reindex...")
            from .utils import reindex_rosbag
            reindexed_file = reindex_rosbag(bag_file)
            if reindexed_file:
                return rosbag.Bag(reindexed_file)
            else:
                return rosbag.Bag(bag_file, 'r', allow_unindexed=True)





