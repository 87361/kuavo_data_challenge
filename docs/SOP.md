真机赛手册  https://kdc-doc.netlify.app/tianchi/cn/pages/real

真机赛手册--submit:https://kdc-doc.netlify.app/tianchi/cn/pages/submit

git clone -b tianchi --depth=1 https://github.com/LejuRobotics/kuavo_data_challenge.git

绝对禁止任何危险的覆写/回退命令 (Zero-Tolerance on Destructive Commands)
永远不再对用户的所有系统和目录执行 git checkout <file>、git reset --hard、git clean -fd 或 rm -rf 等不可逆且会抹除未保存更改的命令。绝不越俎代庖去处理版本控制的丢弃操作。

“先备份后操作”与“大文件强确认”机制 (Defensive Editing & User Approval)
对已有的核心文件进行大范围重构或注入新内容前，先将其内容在上下文中缓存，并在必要时先生成一份 .bak 备份文件后再写入。
如果弄乱文件，如实向汇报并请求用户指导处理方案。

机器说明：
我们现在位于宿主机（笔记本）如果有代码更新请运行同步脚本
1.宿主机（笔记本） 通过ssh连接云开发机，用rsync同步代码，只用于代码开发
2.云开发机-无管理员权限（ssh pi1022）  有权重和环境 云开发机和火山服务器共享文件挂载系统  可以用于测试
开发机 SSH 偶尔有临时网络问题，需要多试几次
远端代码仓库路径 /data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge
远端权重存储在nas盘
同步脚本已经放到 scripts/dev/sync_to_cloud.sh，默认只同步显式传入的文件，或者用 --changed 同步当前改动，且会跳过删除项，避免误删远端。
3.火山服务器-默认管理员root权限  用于训练模型，不要用tmux
4.部署机  用于部署模型，有两张4090

## 真机赛 Task1 smoke 流程（50 条 real_suzhou_3.0，ACT，LeRobot v2.1/v3.0）

手册关键值：分支用 `tianchi`；真机图像 `848x480`；task1 夹爪 `leju_claw`；本次训练不用 depth。

### 1. 仓库分支

```bash
git fetch origin tianchi && git switch tianchi && git pull --ff-only && git submodule update --init --recursive
ssh pi1022 'cd /data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge && git fetch origin tianchi && git switch tianchi && git pull --ff-only && git submodule update --init --recursive'
```

### 2. 数据路径

```bash
RAW=/home/yly/data/kuavo_tianchi/raw/real_suzhou_3.0/task1_zhuomian
V21=/home/yly/data/kuavo_tianchi/lerobot_v21_task1_50/task1_zhuomian
V30=/home/yly/data/kuavo_tianchi/lerobot_v30_task1_50/task1_zhuomian
```

本次已下载 50 条 task1 rosbag：

```bash
find "$RAW" -maxdepth 1 -name '*.bag' | wc -l
du -sh "$RAW"
```

### 3. 转 LeRobot v3.0

```bash
PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 conda run -n kdc_icra python kuavo_data/CvtRosbag2Lerobot.py \
  rosbag.rosbag_dir="$RAW" \
  rosbag.lerobot_dir="$V30" \
  rosbag.chunk_size=100 \
  dataset.eef_type=leju_claw \
  dataset.which_arm=both \
  dataset.use_depth=false \
  dataset.resize.width=848 \
  dataset.resize.height=480 \
  dataset.sample_drop=10 \
  'dataset.task_description=Task1 Desktop Parts Pick And Place'
```

### 4. 转 LeRobot v2.1

```bash
PYTHONPATH=$PWD HYDRA_FULL_ERROR=1 conda run -n kdc python kuavo_data/CvtRosbag2Lerobot.py \
  rosbag.rosbag_dir="$RAW" \
  rosbag.lerobot_dir="$V21" \
  rosbag.chunk_size=100 \
  dataset.eef_type=leju_claw \
  dataset.which_arm=both \
  dataset.use_depth=false \
  dataset.resize.width=848 \
  dataset.resize.height=480 \
  dataset.sample_drop=10 \
  'dataset.task_description=Task1 Desktop Parts Pick And Place'
```

### 5. 校验并上传数据

```bash
python - <<'PY'
import json, pathlib
for p in [
  '/home/yly/data/kuavo_tianchi/lerobot_v21_task1_50/task1_zhuomian/lerobot/meta/info.json',
  '/home/yly/data/kuavo_tianchi/lerobot_v30_task1_50/task1_zhuomian/lerobot/meta/info.json',
]:
    info=json.loads(pathlib.Path(p).read_text())
    print(info['codebase_version'], info['total_episodes'], info['total_frames'], info['fps'])
PY

rsync -avh --info=progress2 /home/yly/data/kuavo_tianchi/lerobot_v21_task1_50/ pi1022:/data/vepfs/users/intern/lingyue.yang/datasets/kuavo_tianchi/lerobot_v21_task1_50/
rsync -avh --info=progress2 /home/yly/data/kuavo_tianchi/lerobot_v30_task1_50/ pi1022:/data/vepfs/users/intern/lingyue.yang/datasets/kuavo_tianchi/lerobot_v30_task1_50/
```

本次结果：`v2.1 50 6376 10`，`v3.0 50 6376 10`。

### 6. 开发机训练 ACT 15 epoch

```bash
ssh pi1022 'cd /data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge && \
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD:$PWD/kuavo_train HYDRA_FULL_ERROR=1 \
conda run -n kdc_icra python kuavo_train/train_policy.py --config-name=act_config \
  task=task1_zhuomian \
  method=act_real50_smoke \
  timestamp=real50_15ep_20260622_135333 \
  repoid=lerobot/task1_zhuomian \
  root=/data/vepfs/users/intern/lingyue.yang/datasets/kuavo_tianchi/lerobot_v30_task1_50/task1_zhuomian/lerobot \
  training.max_epoch=15 \
  training.save_freq_epoch=5 \
  training.batch_size=8 \
  training.num_workers=2 \
  training.scheduler_warmup_steps=10 \
  training.RGB_Augmenter.enable=false \
  policy.custom.use_depth=false \
  policy.use_amp=true \
  policy_name=act'
```

本次训练实测约 `5 min/epoch`；结果在：

```bash
/data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge/outputs/train/task1_zhuomian/act_real50_smoke/run_real50_15ep_20260622_135333
```

校验：

```bash
ssh pi1022 'cd /data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge && \
PYTHONPATH=$PWD:$PWD/kuavo_train conda run -n kdc_icra python -c "from pathlib import Path; from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper; p=Path(\"outputs/train/task1_zhuomian/act_real50_smoke/run_real50_15ep_20260622_135333/epoch15\"); policy=CustomACTPolicyWrapper.from_pretrained(p, strict=True); print(policy.config.n_obs_steps, policy.config.chunk_size)"'
```

### 7. Docker 准备

开发机 `pi1022` 当前用户没有 Docker daemon 权限，只能准备上下文；真正 build/save 在部署机或有 Docker 权限的机器跑。

开发机已准备：

```bash
cd /data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge
conda run -n base conda-pack -n kdc_icra -o myenv.tar.gz --force --ignore-editable-packages
ls -lh myenv.tar.gz outputs/train/task1_zhuomian/act_real50_smoke/run_real50_15ep_20260622_135333/epoch15/model.safetensors
```

有 Docker 权限的机器执行：

```bash
docker build -t kdc_task1_act_smoke .
docker save -o kdc_task1_act_smoke.tar kdc_task1_act_smoke
```

本机已完成：`kdc_task1_act_smoke:latest`，镜像约 `12.7GB`，导出文件 `kdc_task1_act_smoke.tar` 约 `12GB`。

容器内最小校验：

```bash
docker run --rm --net=host kdc_task1_act_smoke bash -lc 'source /opt/ros/noetic/setup.bash && source /root/kuavo_data_challenge/myenv/bin/activate && cd /root/kuavo_data_challenge && PYTHONPATH=$PWD:$PWD/kuavo_train python -c "from pathlib import Path; from kuavo_deploy.config import load_kuavo_config; from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper; cfg=load_kuavo_config(\"configs/deploy/kuavo_real_task1_act_smoke.yaml\"); p=Path(\"outputs/train\")/cfg.inference.task/cfg.inference.method/cfg.inference.timestamp/f\"epoch{cfg.inference.epoch}\"; print(cfg.env.env_name, cfg.env.eef_type, cfg.env.image_size, p.exists()); policy=CustomACTPolicyWrapper.from_pretrained(p, strict=True); print(policy.config.n_obs_steps, policy.config.chunk_size)"'
```

输出应包含：`Kuavo-Real leju_claw [848, 480] True` 和 `1 100`。镜像内已预缓存 `resnet18-f37072fd.pth`，校验时不需要临时下载 backbone 权重。

启动镜像：

```bash
bash docker/run_with_gpu.sh
```

容器内运行 task1 真机配置：

```bash
python kuavo_deploy/src/scripts/script.py --task run --config configs/deploy/kuavo_real_task1_act_smoke.yaml
```
