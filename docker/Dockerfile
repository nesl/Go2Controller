# This is an auto generated Dockerfile for ros:desktop
# generated from docker_images_ros2/create_ros_image.Dockerfile.em
FROM ros:humble-ros-base-jammy

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop=0.10.0-1* \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y cmake gcc build-essential libfftw3-dev libconfig-dev libasound2-dev libpulse-dev libgfortran-*-dev perl

RUN apt-get install -y gfortran texinfo pip && pip install libconf

RUN git clone https://github.com/introlab/audio_utils.git

WORKDIR audio_utils

RUN git submodule update --init --recursive

SHELL ["/bin/bash", "-c"]


RUN apt-get install -y ros-humble-image-tools ros-humble-vision-msgs clang portaudio19-dev

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y

RUN /root/.cargo/bin/rustup install 1.79.0 && /root/.cargo/bin/rustup default 1.79.0

RUN mkdir -p /ros2_ws

WORKDIR /ros2_ws

RUN git clone --recurse-submodules https://github.com/abizovnuralem/go2_ros2_sdk.git src 

WORKDIR src 

RUN pip install -r requirements.txt

WORKDIR ..

RUN source /opt/ros/humble/setup.bash && rosdep install --from-paths src --ignore-src -r -y && colcon build

RUN apt-get install -y gedit

RUN apt-get install -y ffmpeg libavdevice-dev libavfilter-dev libpostproc-dev ros-humble-nav2-costmap-2d ros-humble-nav2-msgs ros-humble-twist-mux ros-humble-navigation2 ros-humble-nav2-amcl ros-humble-nav2-bringup ros-humble-nav2-map-server ros-humble-nav2-util ros-humble-pointcloud-to-laserscan ros-humble-slam-toolbox ros-humble-foxglove-bridge ros-humble-tf-transformations

RUN apt-get install -y nano less wget

#RUN pip install "numpy<1.24" transformers accelerate ultralytics "git+https://github.com/openai/CLIP.git" pydub gtts groq

RUN pip install ultralytics transformers accelerate torchvision faster-whisper soundfile openai resemblyzer gurobipy speechbrain torchaudio

#RUN git clone https://github.com/introlab/odas_ros.git

#RUN rm -r go2_ros2_sdk odas_ros/odas_ros

#RUN mkdir -p /root/.local/share/ov/pkg/

#COPY isaac-sim-2023.1.1 /root/.local/share/ov/pkg/isaac-sim-2023.1.1

COPY docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
