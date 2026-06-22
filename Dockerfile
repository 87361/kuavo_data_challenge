# =========================
# Stage 1: Builder
# =========================
FROM ros:noetic-ros-core-focal AS builder

ARG DEBIAN_FRONTEND=noninteractive

# 国内APT源
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 安装必要工具和ROS依赖
RUN apt-get update && apt-get install -y \
    curl wget gnupg2 lsb-release sudo ca-certificates build-essential bzip2 \
    ros-noetic-cv-bridge \
    ros-noetic-apriltag-ros \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniforge
ENV MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh"
RUN curl -L ${MINIFORGE_URL} -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 配置国内镜像并安装 mamba
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    && conda config --set show_channel_urls yes \
    && conda install -y mamba -c conda-forge

# 工作目录
WORKDIR /root/kuavo_data_challenge
COPY . .

# 解压 Conda 环境并安装项目
RUN if [ -f "myenv.tar.gz" ]; then \
        mkdir -p ./myenv && tar -xzf myenv.tar.gz -C ./myenv && rm myenv.tar.gz; \
    fi && \
    /bin/bash -c "\
        source ./myenv/bin/activate && \
        conda-unpack && \
        pip install -e . && \
        cd ./third_party/lerobot && pip install -e . && \
        pip install deprecated==1.3.1 kuavo_humanoid_sdk==1.3.3 opencv-python==4.12.0.88 opencv-python-headless==4.12.0.88 numpy==2.2.6 && \
        conda clean -afy && \
        rm -rf ./myenv/lib/python*/site-packages/*/tests ./myenv/lib/python*/site-packages/*/test ./myenv/pkgs/* \
    "

RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    curl -L https://download.pytorch.org/models/resnet18-f37072fd.pth \
    -o /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

# =========================
# Stage 2: Final
# =========================
FROM ros:noetic-ros-core-focal

# 设置工作目录
WORKDIR /root/kuavo_data_challenge

# 复制 Conda 环境和项目代码
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/kuavo_data_challenge /root/kuavo_data_challenge
COPY --from=builder /root/.cache/torch/hub/checkpoints /root/.cache/torch/hub/checkpoints

# 环境变量
ENV PATH="/opt/conda/bin:${PATH}"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-apriltag-ros \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 保留 ROS 环境变量
ENV ROS_MASTER_URI=http://kuavo_master:11311
ENV ROS_IP=192.168.26.10

# 激活 Conda 环境
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/kuavo_data_challenge/myenv/bin/activate" >> /root/.bashrc && \
    chmod 777 -R /root/kuavo_data_challenge/kuavo_deploy

# 默认命令
CMD ["bash"]
