调查结果总结
1. Stats文件格式
位置: third_party/vlash/vlash/datasets/compat.py 的 patched_load_stats 函数（190-222行）
格式说明:
v3.0格式: 从 meta/stats.json 加载单个全局stats文件
v2.1格式: 从 meta/episodes_stats.jsonl 加载每episode的stats，然后通过 aggregate_stats 聚合
stats.json预期格式:
{  "feature_name": {    "min": [...],    "max": [...],    "mean": [...],    "std": [...],    "count": [...],    "q01": [...],  // 可选    "q10": [...],  // 可选    "q90": [...],  // 可选    "q99": [...]   // 可选  }}
关键代码:
第203行: v3.0格式检查 (local_dir / STATS_PATH).exists()
第208-209行: v2.1格式如果文件不存在返回 None
第214-215行: 如果episodes_stats为空也返回 None
2. Stats在策略中的初始化
位置: third_party/vlash/vlash/policies/normalize.py 的 create_stats_buffers 函数（56-155行）
默认行为:
当 stats=None 或缺少某个key时，使用 torch.inf 作为默认值
第105-106行: MEAN_STD模式初始化 mean 和 std 为 torch.inf
第116-117行: MIN_MAX模式初始化 min 和 max 为 torch.inf
第128-129行: QUANTILES模式初始化 q01 和 q99 为 torch.inf
第140-141行: QUANTILE10模式初始化 q10 和 q90 为 torch.inf
检查逻辑:
第108行: if key in stats and "mean" in stats[key] and "std" in stats[key]
如果条件不满足，保持 torch.inf 值
问题: 没有检查 torch.inf 的验证代码，也没有warning或error日志
3. Stats传递链路
完整路径:
数据集加载阶段:
   LeRobotDatasetMetadata.__init__()    → load_metadata() (第164行)   → load_stats() (被patched_load_stats替换)   → ds_meta.stats
策略初始化阶段:
   make_policy() (factory.py:86)   → kwargs["dataset_stats"] = ds_meta.stats   → PI0Policy/PI05Policy.__init__() (modeling_pi0.py:853 或 modeling_pi05.py:1122)   → Normalize/Unnormalize.__init__() (normalize.py:174, 257)   → create_stats_buffers() (normalize.py:56)
推理使用阶段:
   policy.unnormalize_outputs.forward() (normalize.py:270)   → 使用buffer中的stats进行反归一化
4. 潜在问题
问题1: stats为None时的行为
patched_load_stats 可能返回 None（第209行、215行）
LeRobotDatasetMetadata.stats 可能为 None（第164行）
传递给策略时，create_stats_buffers 会将所有统计量初始化为 torch.inf
问题2: stats缺少某个key时的行为
如果stats存在但缺少某个feature的key，该feature的统计量会被初始化为 torch.inf
如果stats存在但某个feature缺少必需的统计量（如MEAN_STD模式缺少"mean"或"std"），也会使用 torch.inf
问题3: Unnormalize使用inf时的结果
MEAN_STD模式（第294行）: batch[key] = batch[key] * std + mean
如果 std=inf，结果会是 inf 或 nan
MIN_MAX模式（第301行）: batch[key] = batch[key] * (max_val - min_val) + min_val
如果 min_val=inf 或 max_val=inf，结果会是 inf 或 nan
QUANTILES模式（第309行）: 类似问题
问题4: 缺少错误检测
没有检查 torch.inf 的代码（grep未找到 isinf）
虽然有 no_stats_error_str 函数（第48-53行），但未被调用
没有warning或error日志来指示stats问题
关键代码位置
Stats加载: third_party/vlash/vlash/datasets/compat.py:190-222
Stats初始化: third_party/vlash/vlash/policies/normalize.py:56-155
Unnormalize实现: third_party/vlash/vlash/policies/normalize.py:269-322
策略初始化: third_party/vlash/vlash/policies/factory.py:68-118
数据集metadata: third_party/lerobot/src/lerobot/datasets/lerobot_dataset.py:164
建议
在 create_stats_buffers 中添加检查，当检测到 torch.inf 时发出warning或error
在 patched_load_stats 返回 None 时记录warning
在 Unnormalize.forward() 中添加检查，避免使用inf值进行计算
考虑在策略初始化时验证stats的完整性