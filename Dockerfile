FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
MAINTAINER shiien <shihaosen98@gmail.com>

RUN apt-key del "7fa2af80" \
&& export this_distro="$(cat /etc/os-release | grep '^ID=' | awk -F'=' '{print $2}')" \
&& export this_version="$(cat /etc/os-release | grep '^VERSION_ID=' | awk -F'=' '{print $2}' | sed 's/[^0-9]*//g')" \
&& apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/${this_distro}${this_version}/x86_64/3bf863cc.pub" \
&& apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/machine-learning/repos/${this_distro}${this_version}/x86_64/7fa2af80.pub"

RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list
# RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt update -y && DEBIAN_FRONTEND=noninteractive apt install -y --allow-unauthenticated --no-install-recommends \
    build-essential apt-utils cmake git curl vim ca-certificates \
    libjpeg-dev libpng-dev \
    libgtk3.0 libsm6 cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev \
    libboost-python-dev libtinyxml-dev bash \
    wget unzip libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev openssh-server \
    libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev patchelf libglfw3 zlib1g-dev unrar \
    libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg

RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz && tar -xvf mujoco.tar.gz \
    && mkdir ~/.mujoco && cp -r mujoco210 ~/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
RUN pip install  gym-super-mario-bros==7.3.2 \
    opencv-python future pyglet gym-minigrid -U 'mujoco-py<2.2,>=2.1' gym[atari,box2d] procgen \
    pathlib ray[default] pygame mujoco
RUN wget http://www.atarimania.com/roms/Roms.rar -O roms.rar && unrar x roms.rar \
    && ale-import-roms ROMS/ > ale.out
RUN python -c 'import mujoco_py; import gym'
RUN echo -e "\033[?25h"
RUN rm -r /var/lib/apt/lists/*
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install flax dm-haiku rlax
WORKDIR /HappyResearch
#RUN pip install git+https://github.com/kenjyoung/MinAtar.git


