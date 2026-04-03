# =========================
# Stage 1: Builder
# =========================
FROM ros:noetic-ros-core-focal AS builder

ARG DEBIAN_FRONTEND=noninteractive

#Domestic APT sources
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

#Install necessary tools and ROS dependencies
RUN apt-get update && apt-get install -y \
    curl wget gnupg2 lsb-release sudo ca-certificates build-essential bzip2 \
    ros-noetic-cv-bridge \
    ros-noetic-apriltag-ros \
    && rm -rf /var/lib/apt/lists/*

#Install Miniforge
ENV MINIFORGE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/Release%2025.3.1-0/Miniforge3-25.3.1-0-Linux-x86_64.sh"
RUN curl -L ${MINIFORGE_URL} -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

#Configure domestic mirroring and install mamba
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    && conda config --set show_channel_urls yes \
    && conda install -y mamba -c conda-forge

#working directory
WORKDIR /root/kuavo_data_challenge
COPY . .

#Unzip the Conda environment and install the project
RUN if [ -f "myenv.tar.gz" ]; then \
        mkdir -p ./myenv && tar -xzf myenv.tar.gz -C ./myenv && rm myenv.tar.gz; \
    fi && \
    /bin/bash -c "\
        source ./myenv/bin/activate && \
        conda-unpack && \
        pip install -e . && \
        cd ./third_party/lerobot && pip install -e . && \
        conda clean -afy && \
        rm -rf ./myenv/lib/python*/site-packages/*/tests ./myenv/lib/python*/site-packages/*/test ./myenv/pkgs/* \
    "

# =========================
# Stage 2: Final
# =========================
FROM ros:noetic-ros-core-focal

#Set working directory
WORKDIR /root/kuavo_data_challenge

#Copy the Conda environment and project code
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/kuavo_data_challenge /root/kuavo_data_challenge

#environment variables
ENV PATH="/opt/conda/bin:${PATH}"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-apriltag-ros \
    && rm -rf /var/lib/apt/lists/*

#Preserve ROS environment variables
#Activate the Conda environment
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/kuavo_data_challenge/myenv/bin/activate" >> /root/.bashrc

#Default command
CMD ["bash"]
