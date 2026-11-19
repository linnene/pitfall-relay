#!/bin/bash
# Start the locked RCRS server and the official ADF Java sample team.
set -euo pipefail
D=${D:-/data/amgg_flex_demo/flex-rcrs-transport-pair-0de672a9}
S=$D/src/rcrs-server
A=$D/src/adf-sample
E=$D/envs/rcrs
OUT=${OUT:?set OUT}
MAP=${MAP:-../maps/test/map}
CONFIG=${CONFIG:-../maps/test/config}
mkdir -p "$OUT/logs"
cd "$S/scripts"
export PATH=$E/bin:$PATH
export JAVA_HOME=$E
bash start.sh -m "$MAP" -c "$CONFIG" -g -l "$OUT/logs" \
  > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$OUT/server.pid"
for _ in $(seq 1 120); do
  grep -q "Start your agents" "$OUT/server.log" && break
  sleep 1
done
if ! grep -q "Start your agents" "$OUT/server.log"; then
  echo "server did not reach agent phase" >&2
  exit 1
fi
cd "$A"
export GRADLE_USER_HOME=/data/amgg_flex_demo/flex-rcrs-feasibility-6a2f1d8c/gradle
/data/amgg_flex_demo/flex-rcrs-feasibility-6a2f1d8c/src/gradle-9.3.1/bin/gradle launch \
  --args="adf.impl.DefaultLoader -t 1,0,1,0,1,0 -h localhost -p 27931" \
  --no-daemon > "$OUT/adf.log" 2>&1 &
ADF_PID=$!
echo "$ADF_PID" > "$OUT/adf.pid"
echo "server=$SERVER_PID adf=$ADF_PID"
wait $SERVER_PID || true
kill "$ADF_PID" 2>/dev/null || true
echo "server finished"
