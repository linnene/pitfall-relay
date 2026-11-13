FROM isaacsim4.5.0-rosbridge:local

# Nav2 for Humble plus python3-lark, which ros2launch needs but the Isaac ROS
# bridge base image does not pull in. libbrotli1 is pinned because the base
# image carries a newer third-party brotli that conflicts with libbrotli-dev
# pulled by the ROS desktop dependency chain.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades \
        libbrotli1=1.0.9-2ubuntu0.1 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-tf2-tools \
        ros-humble-xacro \
        python3-lark && \
    rm -rf /var/lib/apt/lists/*
