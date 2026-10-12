"""Calibrate the camera's horizontal field of view for the geometric pre-position estimate.

`pick_place.py`'s geometric pre-position needs to convert a pixel offset into
a real-world mm offset on the floor plane. That conversion depends on the
camera's field of view, which we were previously just guessing (60 deg).

Method: place a ruler (or anything with a known length) flat on the table,
in view of the camera, at the same height you'll actually be picking from.
Click its two end points on the image, tell this script the real distance
between them in mm, and it solves for the field of view:

  mm_per_pixel = real_mm / pixel_distance
  tan(HFOV/2)  = mm_per_pixel * frame_width / (2 * height_above_floor)
  HFOV         = 2 * atan(...)

Requires floor_calibration.json (run calibrate_floor.py first) so we know
how high the camera is above the table right now.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2

from roarm_driver import RoArm

OUT_PATH = "camera_calibration.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate camera horizontal FOV via a known-length reference.")
    parser.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    parser.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--floor-calibration", default="floor_calibration.json")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    if not Path(args.floor_calibration).exists():
        print(f"{args.floor_calibration} not found -- run calibrate_floor.py first (need the floor height).")
        return 1
    floor_z = float(json.loads(Path(args.floor_calibration).read_text(encoding="utf-8"))["z"])

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)
    pose = arm.feedback()
    if pose is None:
        print("could not read arm feedback, check the connection.")
        return 1
    height_mm = pose.z - floor_z
    if height_mm <= 5:
        print(f"height above floor is only {height_mm:.1f}mm -- move the arm up first, then rerun.")
        return 1
    print(f"camera height above floor: {height_mm:.1f}mm")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"cannot open camera index {args.camera_index}")
        return 1

    window = "FOV calibration - click two ends of a known-length object, then press c"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    print("자를 테이블(집을 위치와 같은 높이)에 평평하게 놓고, 두 끝점을 클릭하세요. 확정: c, 취소: q")

    frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            for i, (x, y) in enumerate(points):
                cv2.circle(frame, (x, y), 6, (0, 220, 0), -1)
                cv2.putText(frame, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], (0, 220, 0), 2)
            cv2.imshow(window, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("취소합니다.")
                return 1
            if key == ord("c") and len(points) == 2:
                break
    finally:
        cap.release()
        cv2.destroyWindow(window)

    (x1, y1), (x2, y2) = points
    pixel_distance = math.hypot(x2 - x1, y2 - y1)
    if pixel_distance < 2:
        print("두 점이 너무 가깝습니다. 다시 실행하세요.")
        return 1

    real_mm = float(input("두 점 사이의 실제 거리(mm)를 입력하세요: ").strip())

    frame_width = frame.shape[1]
    mm_per_pixel = real_mm / pixel_distance
    half_hfov = math.atan(mm_per_pixel * frame_width / (2 * height_mm))
    hfov_deg = math.degrees(2 * half_hfov)

    print(f"pixel_distance={pixel_distance:.1f}px, mm_per_pixel={mm_per_pixel:.3f}, HFOV={hfov_deg:.1f}deg")

    data = {
        "hfov_deg": hfov_deg,
        "height_mm_at_calibration": height_mm,
        "mm_per_pixel_at_calibration": mm_per_pixel,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장됨: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
