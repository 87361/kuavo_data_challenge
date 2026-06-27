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

我的modelscope数据集：Xyyyhamster/robo_data_0626
#上传本地处理好的数据目录
ms upload Xyyyhamster/robo_data_0626 /path/to/processed_data --repo-type dataset --commit-message "upload processed dataset"

## 数据集清洗工具：
端口清理+启动：
```bash
ss -ltnp 'sport = :18080 or sport = :18081'
ss -ltnp 'sport = :18080 or sport = :18081' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | xargs -r kill
python scripts/gui/lerobot_editor/server.py --host 127.0.0.1 --port 18080
```
按照特定条件提取特定数据并转换为v3.0的脚本：
scripts/data/export_scored_lerobot_v30.py
默认用法：
python scripts/data/export_scored_lerobot_v30.py \
  --source-root /path/to/edited_v21/lerobot \
  --output-root /path/to/output_v30/task/lerobot

## 真机赛 Task1 smoke 流程（50 条 real_suzhou_3.0，ACT，LeRobot v2.1/v3.0）

手册关键值：分支用 `tianchi`；真机图像 `848x480`；task1 夹爪 `leju_claw`；本次训练不用 depth。

### 1. 仓库分支

```bash
git fetch origin tianchi && git switch tianchi && git pull --ff-only && git submodule update --init --recursive
ssh pi1022 'cd /data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge && git fetch origin tianchi && git switch tianchi && git pull --ff-only && git submodule update --init --recursive'
```

### 2. 数据路径

```bash
DATA_ROOT=/home/yly/data/kuavo_tianchi
RAW_ROOT=$DATA_ROOT/raw
RAW=$RAW_ROOT/real_suzhou_3.0/task1_zhuomian
V21=/home/yly/data/kuavo_tianchi/lerobot_v21_task1_50/task1_zhuomian
V30=/home/yly/data/kuavo_tianchi/lerobot_v30_task1_50/task1_zhuomian
REMOTE_DATA=/data/vepfs/users/intern/lingyue.yang/datasets/kuavo_tianchi
```

ModelScope 下载：

```bash
python -m pip install -U modelscope pandas pyarrow
mkdir -p "$RAW_ROOT"
modelscope download --dataset lejurobot/LET-Tianchi-Dataset \
  --include 'real_suzhou_3.0/task1_zhuomian/*.bag' \
  --local_dir "$RAW_ROOT" \
  --max-workers 8
```

只拉前 50 条：

```bash
mkdir -p "$RAW_ROOT"
python - <<'PY' > /tmp/kdc_task1_50_files.txt
from modelscope.hub.api import HubApi
fs=HubApi().get_dataset_files('lejurobot/LET-Tianchi-Dataset', root_path='real_suzhou_3.0/task1_zhuomian', recursive=True, page_size=1000)
print('\n'.join(sorted(f['Path'] for f in fs if f.get('Path','').endswith('.bag'))[:50]))
PY
xargs -a /tmp/kdc_task1_50_files.txt modelscope download \
  --dataset lejurobot/LET-Tianchi-Dataset \
  --local_dir "$RAW_ROOT" \
  --max-workers 8
```

校验：

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

ssh pi1022 "mkdir -p $REMOTE_DATA"

rsync -avh --info=progress2 \
  /home/yly/data/kuavo_tianchi/lerobot_v21_task1_50/ \
  pi1022:$REMOTE_DATA/lerobot_v21_task1_50/

rsync -avh --info=progress2 \
  /home/yly/data/kuavo_tianchi/lerobot_v30_task1_50/ \
  pi1022:$REMOTE_DATA/lerobot_v30_task1_50/
```

上传 raw 到开发机：

```bash
ssh pi1022 "mkdir -p $REMOTE_DATA/raw/real_suzhou_3.0/task1_zhuomian"
rsync -avh --info=progress2 \
  /home/yly/data/kuavo_tianchi/raw/real_suzhou_3.0/task1_zhuomian/ \
  pi1022:$REMOTE_DATA/raw/real_suzhou_3.0/task1_zhuomian/
```

拉回转换数据：

```bash
mkdir -p /home/yly/data/kuavo_tianchi
rsync -avh --info=progress2 \
  pi1022:$REMOTE_DATA/lerobot_v21_task1_50/ \
  /home/yly/data/kuavo_tianchi/lerobot_v21_task1_50/
rsync -avh --info=progress2 \
  pi1022:$REMOTE_DATA/lerobot_v30_task1_50/ \
  /home/yly/data/kuavo_tianchi/lerobot_v30_task1_50/
```

本次结果：`v2.1 50 6376 10`，`v3.0 50 6376 10`。

## 真机赛 Task1 全量数据准备（1000 条 real_suzhou_3.0，LeRobot v2.1/v3.0）

`/home` 当前空间不够，全量数据放 `/mnt/data`。2026-06-23 元数据检查：`real_suzhou_3.0/task1_zhuomian` 共 `1000` 个 `.bag`，raw 约 `486.29 GiB`。

默认流程：源 rosbag 先下载到本机 `$DATA_ROOT/raw`，每批转换成 v2.1 后删除该批 raw 以节省空间；最后合并完整 v2.1，再从 v2.1 转出完整 v3.0，并上传开发机。中断后重跑同一条 `run` 命令即可从状态目录继续。

```bash
DATA_ROOT=/mnt/data/kuavo_tianchi
REMOTE_DATA=/data/vepfs/users/intern/lingyue.yang/datasets/kuavo_tianchi

python -m pip install -U modelscope pandas pyarrow

python scripts/data/prepare_task1_full_dataset.py plan \
  --data-root "$DATA_ROOT" \
  --batch-max-gib 80

python scripts/data/prepare_task1_full_dataset.py run \
  --data-root "$DATA_ROOT" \
  --batch-max-gib 80 \
  --max-workers 8 \
  --v21-env kdc \
  --v30-env kdc_icra \
  --upload \
  --remote pi1022:$REMOTE_DATA
```

如果要长期保留完整 raw，本机 `/mnt/data` 理论上够但余量较紧，建议减小批次：

```bash
python scripts/data/prepare_task1_full_dataset.py run \
  --data-root /mnt/data/kuavo_tianchi \
  --batch-max-gib 40 \
  --keep-raw \
  --max-workers 8 \
  --v21-env kdc \
  --v30-env kdc_icra \
  --upload \
  --remote pi1022:/data/vepfs/users/intern/lingyue.yang/datasets/kuavo_tianchi
```

只校验本机转换结果：

```bash
python scripts/data/prepare_task1_full_dataset.py verify \
  --data-root /mnt/data/kuavo_tianchi
```

输出路径：

```bash
/mnt/data/kuavo_tianchi/lerobot_v21_task1_full/task1_zhuomian/lerobot
/mnt/data/kuavo_tianchi/lerobot_v30_task1_full/task1_zhuomian/lerobot
pi1022:$REMOTE_DATA/lerobot_v21_task1_full/task1_zhuomian/lerobot
pi1022:$REMOTE_DATA/lerobot_v30_task1_full/task1_zhuomian/lerobot
```

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

拉回 Docker build 文件：

```bash
rsync -avh --info=progress2 \
  pi1022:/data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge/myenv.tar.gz \
  ./myenv.tar.gz
mkdir -p outputs/train/task1_zhuomian/act_real50_smoke
rsync -avh --info=progress2 \
  pi1022:/data/vepfs/users/intern/lingyue.yang/kuavo_data_challenge/outputs/train/task1_zhuomian/act_real50_smoke/run_real50_15ep_20260622_135333/ \
  outputs/train/task1_zhuomian/act_real50_smoke/run_real50_15ep_20260622_135333/
```

有 Docker 权限的机器执行：

```bash
docker build -t kdc_task1_act_smoke .
docker save -o kdc_task1_act_smoke.tar kdc_task1_act_smoke
```

本机已完成：`kdc_task1_act_smoke:latest`，镜像约 `12.7GB`，导出文件 `kdc_task1_act_smoke.tar` 约 `12GB`。

容器内最小校验：

```bash
docker run --rm -i --net=host kdc_task1_act_smoke bash <<'SH'
source /opt/ros/noetic/setup.bash
source /root/kuavo_data_challenge/myenv/bin/activate
cd /root/kuavo_data_challenge
PYTHONPATH=$PWD:$PWD/kuavo_train:${PYTHONPATH:-} python - <<'PY'
from pathlib import Path
from kuavo_deploy.config import load_kuavo_config
from kuavo_train.wrapper.policy.act.ACTPolicyWrapper import CustomACTPolicyWrapper

cfg = load_kuavo_config("configs/deploy/kuavo_env.yaml")
assert cfg.env.env_name == "Kuavo-Real"
assert cfg.env.which_arm == "right"
assert cfg.env.eef_type == "leju_claw"
assert list(cfg.env.image_size) == [848, 480]
assert cfg.env.obs_key_map["joint_q"]["handle"]["params"]["slice"] == [[12, 19], [19, 26]]
assert cfg.env.obs_key_map["gripper"]["handle"]["params"]["slice"] == [[0, 1], [1, 2]]

p = Path("outputs/train") / cfg.inference.task / cfg.inference.method / cfg.inference.timestamp / f"epoch{cfg.inference.epoch}"
assert p.exists() and (p / "model.safetensors").exists()
policy = CustomACTPolicyWrapper.from_pretrained(p, strict=True)
assert policy.config.n_obs_steps == 1
assert policy.config.chunk_size == 100
print("config_policy_ok", cfg.env.env_name, cfg.env.which_arm, cfg.env.eef_type, cfg.env.image_size)
PY
SH
```

输出应包含：`config_policy_ok Kuavo-Real right leju_claw [848, 480]`。镜像内已预缓存 `resnet18-f37072fd.pth`，校验时不需要临时下载 backbone 权重。

FAQ 对部署推理缺 `pyaudio` 的说明：镜像/环境里需要先有 `portaudio19-dev` 等系统依赖，再安装 `pyaudio`。提交前额外检查 SDK 导入链路：

```bash
docker run --rm --net=host kdc_task1_act_smoke bash -lc 'source /opt/ros/noetic/setup.bash && source /root/kuavo_data_challenge/myenv/bin/activate && cd /root/kuavo_data_challenge && PYTHONPATH=$PWD:$PWD/kuavo_train:${PYTHONPATH:-} python -c "import pyaudio, msgpack, websockets, zmq; from kuavo_humanoid_sdk import KuavoSDK, KuavoRobot, KuavoRobotState, DexterousHand; from kuavo_deploy.kuavo_env.KuavoBaseRosEnv import KuavoBaseRosEnv; print(\"sdk_import_ok\")"'
```

提交前踩坑清单：

- 真机部署配置必须写在 `configs/deploy/kuavo_env.yaml`，`run_with_gpu.sh` 的默认 `KDC_CONFIG` 也必须指向这个文件。
- Task1 是右手单手任务，`kuavo_env.yaml` 里必须是 `which_arm: right`；如果沿用双臂 ACT checkpoint，保留 `policy_which_arm: both` 只用于兼容模型的 16 维 state/action，执行时会只取右臂动作。
- 依赖必须能导入：`pyaudio`、`msgpack`、`websockets`、`zmq`；其中 `pyaudio` 需要系统包 `portaudio19-dev`，`zmq` 由 `pyzmq` 提供。
- 提交 zip 根目录只能有 `kdc_task1_act_smoke.tar` 和 `run_with_gpu.sh` 两个文件。

### 8. 打 zip 并上传 OSS
!!!!!!
注意修改 run_with_gpu.sh 中的docker镜像名称为你刚刚打包的真机赛镜像名称
！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
https://kdc-doc.netlify.app/tianchi/cn/pages/faq  看一下手册里的常见问题，对比一下，再提交
！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

提交 zip 根目录必须有两个文件：`kdc_task1_act_smoke.tar` 和 `run_with_gpu.sh`。

```bash
test ! -f kdc_task1_act_smoke.zip || mv kdc_task1_act_smoke.zip kdc_task1_act_smoke.zip.bak.$(date +%Y%m%d_%H%M%S)
mkdir -p dist/kdc_task1_act_smoke_submit

docker save -o dist/kdc_task1_act_smoke_submit/kdc_task1_act_smoke.tar kdc_task1_act_smoke:latest

cat > dist/kdc_task1_act_smoke_submit/run_with_gpu.sh <<'SH'
#!/bin/bash
set -e
IMAGE_NAME="${IMAGE_NAME:-kdc_task1_act_smoke}"
CONTAINER_NAME="${CONTAINER_NAME:-$IMAGE_NAME}"
IMAGE_TAR="${IMAGE_TAR:-${IMAGE_NAME}.tar}"
KDC_CONFIG="${KDC_CONFIG:-configs/deploy/kuavo_env.yaml}"
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then docker rm -f "${CONTAINER_NAME}"; fi
if [ "$(docker images -q "${IMAGE_NAME}")" ]; then docker rmi -f "${IMAGE_NAME}"; fi
docker load -i "${IMAGE_TAR}"
docker run --gpus all -it --net=host \
  -e ROS_MASTER_URI=http://kuavo_master:11311 \
  -e ROS_IP=192.168.26.10 \
  -e KDC_CONFIG="${KDC_CONFIG}" \
  --name "${CONTAINER_NAME}" \
  "${IMAGE_NAME}" bash
SH
chmod +x dist/kdc_task1_act_smoke_submit/run_with_gpu.sh

(cd dist/kdc_task1_act_smoke_submit && zip -0 -T ../../kdc_task1_act_smoke.zip kdc_task1_act_smoke.tar run_with_gpu.sh)
unzip -l kdc_task1_act_smoke.zip
```

期望 zip 内容只有这两个文件：

```text
kdc_task1_act_smoke.tar
run_with_gpu.sh
```

Linux/Mac 模板：

```bash
./ossutil cp localFilePath oss://${bucket}/${path}/ossFileName -i ${accessKeyId} -k ${accessKeySecret} --endpoint=${endpoint} --sts-token=${securityToken}
```

实际上传命令不要把 AK/SK/token 写入仓库，实时复制到环境变量后执行：
```bash
export OSS_AK='实时获取的 accessKeyId'
export OSS_SK='实时获取的 accessKeySecret'
export OSS_STS_TOKEN='实时获取的 securityToken'
export OSS_ENDPOINT='oss-cn-hangzhou.aliyuncs.com'
export OSS_URI='oss://tianchi-race-upload/result/race/532415/1754/1313451/1095280909904/1782216775492_kdc_task1_act_smoke.zip'

ossutil cp kdc_task1_act_smoke.zip "$OSS_URI" -f \
  -i "$OSS_AK" \
  -k "$OSS_SK" \
  --endpoint="$OSS_ENDPOINT" \
  --sts-token="$OSS_STS_TOKEN" \
  --checkpoint-dir=/tmp/ossutil-checkpoint-kdc-task1
```
注意：AK 和 token 有效期约 1 小时，上传前实时获取；多次上传使用相同 `OSS_URI` 覆盖。部分 STS 只允许上传，可能不允许 `stat/ls` 查询。

启动镜像：

```bash
bash docker/run_with_gpu.sh
```

容器内运行 task1 真机配置：

```bash
python kuavo_deploy/src/scripts/script.py --task run --config configs/deploy/kuavo_env.yaml
```
