"""Render an acceptance summary JSON as a compact Markdown evidence sheet."""

import json
import sys


def render_round(rnd):
    lines = [f"### {rnd['label']}", "", "| robot | goal | status | final (x,y,yaw) | xy err (m) | feedback | cmd_vel | plans |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for name, r in rnd["robots"].items():
        px, py, pyaw = r["final_pose_world"]
        lines.append(
            f"| {name} | {r['goal']} | {r['status']} | ({px}, {py}, {pyaw}) | "
            f"{r['xy_error_m']} | {r['feedback_samples']} | {r['cmd_vel_msgs']} | {r['plan_msgs']} |"
        )
    if rnd.get("timed_out"):
        lines.append(f"\nTimed out: {rnd['timed_out']}")
    return "\n".join(lines)


def main(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"# {data['mode']} run")
    print()
    print(f"map: `{data['map_yaml']}`")
    print()
    for rnd in data["rounds"]:
        print(render_round(rnd))
        print()
    for event in data["events"]:
        if event["event"] in ("probe", "route_blocked", "dynamic_obstacle_inserted"):
            print("```json")
            print(json.dumps(event, ensure_ascii=False))
            print("```")
    if isinstance(data.get("result"), dict) and "interfaces" in data["result"]:
        print("\n## Case 4 interfaces")
        for name, itf in data["result"]["interfaces"].items():
            print(f"- {name}: result={itf['action_result']} odom={itf['odom_msgs']} "
                  f"scan={itf['scan_msgs']} cmd_vel={itf['cmd_vel_msgs']} plan={itf['plan_msgs']}")
        print("\nCross-talk:")
        print("```json")
        print(json.dumps(data["result"]["cross_talk"], indent=2))
        print("```")


if __name__ == "__main__":
    main(sys.argv[1])
