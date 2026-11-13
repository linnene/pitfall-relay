#!/bin/bash
# Entry point inside the isolated Nav2 acceptance container.
set -e
export PYTHONPATH=/work:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages
export LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
export PYTHONDONTWRITEBYTECODE=1
exec /isaac-sim/python.sh /work/scripts/acceptance.py "$@"
