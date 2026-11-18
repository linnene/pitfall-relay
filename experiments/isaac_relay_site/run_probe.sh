#!/bin/bash
# Container entry point for the relay-site development probes.
set -e
export PYTHONPATH=/work:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages
export LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-79}
export PYTHONDONTWRITEBYTECODE=1
exec /isaac-sim/python.sh /work/scripts/bay_probe.py "$@"
