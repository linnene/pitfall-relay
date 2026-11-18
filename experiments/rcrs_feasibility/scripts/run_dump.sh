#!/bin/bash
# Start the official RCRS server on the test map and connect dump agents.
set -euo pipefail
D=${D:-/data/amgg_flex_demo/flex-rcrs-feasibility-6a2f1d8c}
S=$D/src/rcrs-server
E=$D/envs/rcrs
OUT=${OUT:-$D/runs/dump1}
MAX_STEPS=${MAX_STEPS:-40}
mkdir -p "$OUT/logs"

cd "$S/scripts"
export PATH=$E/bin:$PATH
export JAVA_HOME=$E
bash start.sh -m ../maps/test/map -c ../maps/test/config -g -l "$OUT/logs" \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server.pid"

for _ in $(seq 1 120); do
  if grep -q "Start your agents" "$OUT/server.log"; then break; fi
  sleep 1
done
if ! grep -q "Start your agents" "$OUT/server.log"; then
  echo "server did not reach agent phase" >&2
  exit 1
fi

cd "$D/src/rcrs-sample-agent-python"
export PYTHONPATH="$D/src/rcrs-core-python:$D/src/rcrs-sample-agent-python"
for role in POLICE_FORCE AMBULANCE_TEAM FIRE_BRIGADE; do
  ROLE=$role OUT_PATH="$OUT/dump_${role}.json" MAX_STEPS="$MAX_STEPS" \
    "$E/bin/python" dump_agent.py > "$OUT/logs/agent_${role}.log" 2>&1 &
  echo $! >> "$OUT/agents.pid"
done

for _ in $(seq 1 150); do
  n=$(ls "$OUT"/dump_*.json 2>/dev/null | wc -l)
  if [ "$n" -ge 3 ]; then break; fi
  sleep 1
done

while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$OUT/agents.pid"
cd "$S/scripts"
bash kill.sh >/dev/null 2>&1 || true
kill "$SERVER_PID" 2>/dev/null || true
echo "dump complete: $(ls "$OUT"/dump_*.json 2>/dev/null | wc -l) files"
