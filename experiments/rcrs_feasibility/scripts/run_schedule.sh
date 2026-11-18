#!/bin/bash
# One scheduled clear/transport run on the official test map.
set -euo pipefail
D=${D:-/data/amgg_flex_demo/flex-rcrs-feasibility-6a2f1d8c}
S=$D/src/rcrs-server
E=$D/envs/rcrs
OUT=${OUT:?set OUT}
AMB_START=${AMB_START:-3}
MAX_WAIT=${MAX_WAIT:-180}
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
ROLE=POLICE_FORCE OUT_PATH="$OUT/police.json" START_STEP=3 \
  "$E/bin/python" relay_agents.py > "$OUT/logs/police.log" 2>&1 &
echo $! >> "$OUT/agents.pid"
ROLE=FIRE_BRIGADE OUT_PATH="$OUT/fire.json" START_STEP=3 \
  "$E/bin/python" relay_agents.py > "$OUT/logs/fire.log" 2>&1 &
echo $! >> "$OUT/agents.pid"
ROLE=AMBULANCE_TEAM OUT_PATH="$OUT/ambulance.json" START_STEP="$AMB_START" \
  "$E/bin/python" relay_agents.py > "$OUT/logs/ambulance.log" 2>&1 &
echo $! >> "$OUT/agents.pid"

for _ in $(seq 1 "$MAX_WAIT"); do
  if [ -f "$OUT/ambulance.json" ]; then break; fi
  sleep 1
done

while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$OUT/agents.pid"
cd "$S/scripts"
bash kill.sh >/dev/null 2>&1 || true
kill "$SERVER_PID" 2>/dev/null || true
echo "run complete: ambulance=$(test -f "$OUT/ambulance.json" && echo yes || echo no) police=$(test -f "$OUT/police.json" && echo yes || echo no)"
