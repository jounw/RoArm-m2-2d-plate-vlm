"""Teach the place location by hand.

Torque off, physically move the gripper to where you want objects dropped,
press Enter, and the pose (x, y, z, t) is saved to place_pose.json. pick_place.py
then carries every picked object to exactly that spot.

Run: python3 teach_place.py --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from roarm_driver import RoArm

OUT_PATH = "place_pose.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Hand-teach the place location and save it.")
    parser.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    parser.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)

    print("토크를 끕니다. 팔을 손으로 자유롭게 움직일 수 있습니다.")
    arm.torque(False)
    time.sleep(0.3)

    try:
        input("그리퍼를 '놓을 위치'에 맞춘 뒤 Enter를 누르세요 (Ctrl+C로 취소): ")
    except KeyboardInterrupt:
        print("\n취소합니다.")
        arm.torque(True)
        return 1

    pose = arm.feedback()
    if pose is None:
        print("피드백을 읽지 못했습니다. 연결을 확인하세요.")
        arm.torque(True)
        return 1

    print(f"측정된 놓을 위치: x={pose.x:.1f}, y={pose.y:.1f}, z={pose.z:.1f}, t={pose.t:.3f}")

    print("토크를 다시 켭니다.")
    arm.torque(True)
    time.sleep(0.3)

    data = {
        "base_pose": {"x": pose.x, "y": pose.y, "z": pose.z, "t": pose.t},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "serial_port": args.serial_port,
    }
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장됨: {args.out}")
    print("이제 pick_place.py가 집은 물체를 매번 이 위치에 내려놓습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
