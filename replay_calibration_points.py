"""Replay test: move the RoArm sequentially to the N robot-frame points
recorded in lidar_extrinsic.json (from calibrate_lidar_extrinsic.py).

Useful right after calibration to sanity-check the recorded robot_points_mm
are all safely reachable and roughly where you expect, and as a physical
gut-check that the arm actually revisits the same taught positions.

This does NOT touch the camera or the fitted rotation/translation -- it just
replays the raw robot_points_mm that were captured by hand-moving the arm
during calibration.

Run:
  python3 replay_calibration_points.py --serial-port /dev/cu.usbserial-120
  python3 replay_calibration_points.py --serial-port /dev/cu.usbserial-120 --pause 2 --speed 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from roarm_driver import Pose, RoArm


def load_points(path: str) -> list[list[float]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pts = data.get("robot_points_mm")
    if not pts:
        raise ValueError(f"{path} 안에 robot_points_mm이 없습니다 -- calibrate_lidar_extrinsic.py를 먼저 실행하세요.")
    return pts


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the N calibration robot points sequentially, as a reachability/sanity test.")
    parser.add_argument("--extrinsic", default="lidar_extrinsic.json")
    parser.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    parser.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "/dev/cu.usbserial-120"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--speed", type=float, default=0.3, help="move speed (0-1)")
    parser.add_argument("--pause", type=float, default=1.5, help="seconds to hold at each point before moving on")
    parser.add_argument("--gripper", type=float, default=RoArm.GRIPPER_OPEN, help="gripper radian to hold throughout (default: open)")
    parser.add_argument("--start-pose", default="roarm_start_pose.json", help="if present, moves here first/last for a safe start/end position")
    parser.add_argument("--confirm-each", action="store_true", help="wait for Enter before each move instead of just pausing")
    args = parser.parse_args()

    points = load_points(args.extrinsic)
    print(f"{args.extrinsic}에서 {len(points)}개 지점을 불러왔습니다.")

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)

    start_pose = None
    if Path(args.start_pose).exists():
        p = json.loads(Path(args.start_pose).read_text(encoding="utf-8"))["base_pose"]
        start_pose = Pose(x=float(p["x"]), y=float(p["y"]), z=float(p["z"]), t=float(p["t"]))
        print(f"시작 자세로 먼저 이동: x={start_pose.x:.1f} y={start_pose.y:.1f} z={start_pose.z:.1f}")
        arm.move_to(start_pose, speed=args.speed)
        time.sleep(args.pause)

    try:
        for i, (x, y, z) in enumerate(points, start=1):
            print(f"--- 지점 {i}/{len(points)}: x={x:.1f} y={y:.1f} z={z:.1f} ---")
            if args.confirm_each:
                input("  Enter를 누르면 이동합니다 (Ctrl+C 취소): ")
            arm.move_to(Pose(x=x, y=y, z=z, t=args.gripper), speed=args.speed)
            time.sleep(args.pause)
            fb = arm.feedback()
            if fb is not None:
                err = ((fb.x - x) ** 2 + (fb.y - y) ** 2 + (fb.z - z) ** 2) ** 0.5
                print(f"  도착 후 피드백: x={fb.x:.1f} y={fb.y:.1f} z={fb.z:.1f} (목표와 거리차 {err:.1f}mm)")
            else:
                print("  피드백을 읽지 못했습니다.")
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        if start_pose is not None:
            print("시작 자세로 복귀합니다.")
            arm.move_to(start_pose, speed=args.speed)

    print("완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
