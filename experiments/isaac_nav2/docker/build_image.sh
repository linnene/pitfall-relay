#!/bin/bash
# Build the derived Nav2 image on the experiment host (amgg).
set -euo pipefail
cd "$(dirname "$0")"
docker build -t isaacsim4.5.0-nav2:local .
