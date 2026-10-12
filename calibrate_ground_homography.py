"""Calibrate a fixed (non eye-in-hand) camera against the floor plane.

For a camera bolted in a fixed spot (e.g. the wide/background camera on
index 1), the geometric pre-position math in pick_place.py (which assumes
the camera moves with the gripper) doesn't apply. Instead we need a direct
pixel -> real-world(x, y) mapping on the floor plane: a homography.

Procedure:
  1. Mark 4 points on the floor within the camera's view (tape, dots, whatever).
  2. This script shows the live feed; click each of the 4 marked points on the
     image, in any order, confirming each with a key press.
  3. For each point, torque is turned off so you can physically move the
     gripper tip to sit exactly on that marked point, then press Enter to
     record the arm's real (x, y) there.
  4. Repeat for all 4 points, then the script solves the homography
     (pixel -> real x/y on the floor) with cv2.findHomography and saves it.

pick_place.py or a new fixed-camera variant can then load this file and use
cv2.perspectiveTransform (or the manual 3x3 multiply) to convert any pixel
(e.g. a detected object's center) directly into real x/y on the floor --
no field-of-view/height math needed, since the mapping is baked in.

Run: python3 calibrate_ground_homography.py --camera-index 1 --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from roarm_driver import RoArm

OUT_PATH = "ground_homography.json"
NUM_POINTS = 4


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate pixel->floor(x,y) homography for a fixed camera.")
    parser.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    parser.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--camera-index", type=int, default=1, help="the fixed background camera's index")
    parser.add_argument("--num-points", type=int, default=NUM_POINTS, help="how many floor points you marked (>=4)")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    if args.num_points < 4:
        print("호모그래피 계산에는 최소 4개의 점이 필요합니다.")
        return 1

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"cannot open camera index {args.camera_index}")
        return 1

    window = f"Ground calibration (camera {args.camera_index}) - click a marked point, then press c"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear()
            clicked.append((x, y))

    cv2.setMouseCallback(window, on_mouse)

    pixel_points: list[tuple[int, int]] = []
    world_points: list[tuple[float, float]] = []

    print(f"바닥에 표시해둔 점 {args.num_points}개를 하나씩 처리합니다.")
    print("각 점마다: 1) 화면에서 그 점을 클릭  2) 'c'로 확정  3) 팔을 손으로 그 점 위로 옮기고 Enter\n")

    try:
        for i in range(args.num_points):
            print(f"--- 점 {i + 1}/{args.num_points} ---")
            clicked.clear()
            frame = None
            # step 1: click the point on the live image
            while True:
                ok, live = cap.read()
                if ok:
                    frame = live.copy()
                    if clicked:
                        cv2.circle(frame, clicked[0], 7, (0, 220, 0), -1)
                    cv2.putText(
                        frame, f"point {i + 1}/{args.num_points}: click it, then press c (q: cancel)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                    )
                    cv2.putText(
                        frame, f"point {i + 1}/{args.num_points}: click it, then press c (q: cancel)",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    )
                    cv2.imshow(window, frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    print("취소합니다.")
                    return 1
                if key == ord("c") and clicked:
                    break
            px, py = clicked[0]
            print(f"  화면 좌표: ({px}, {py})")

            # step 2: hand-move the arm onto that same physical point
            print("  토크를 끕니다. 그리퍼 끝을 방금 클릭한 표시점 위로 옮기세요.")
            arm.torque(False)
            time.sleep(0.3)
            try:
                input("  준비되면 Enter (Ctrl+C로 취소): ")
            except KeyboardInterrupt:
                print("\n취소합니다.")
                arm.torque(True)
                return 1

            pose = arm.feedback()
            if pose is None:
                print("  피드백을 읽지 못했습니다. 연결을 확인하세요.")
                arm.torque(True)
                return 1
            print(f"  측정된 실제 위치: x={pose.x:.1f}, y={pose.y:.1f}, z={pose.z:.1f}")

            pixel_points.append((px, py))
            world_points.append((pose.x, pose.y))

        print("토크를 다시 켭니다.")
        arm.torque(True)
        time.sleep(0.3)
    finally:
        cap.release()
        cv2.destroyWindow(window)

    src = np.array(pixel_points, dtype=np.float64)
    dst = np.array(world_points, dtype=np.float64)

    if len(pixel_points) == 4:
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    else:
        H, _ = cv2.findHomography(src, dst, method=0)

    if H is None:
        print("호모그래피를 계산하지 못했습니다 (점들이 일직선에 가깝지 않은지 확인하세요).")
        return 1

    data = {
        "camera_index": args.camera_index,
        "homography": H.tolist(),  # 3x3: pixel(x,y,1) -> world(x,y,w), then divide by w
        "pixel_points": pixel_points,
        "world_points": world_points,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장됨: {args.out}")
    print("사용법: 픽셀 (px, py) -> [X, Y, W] = H @ [px, py, 1]; 실제 좌표 = (X/W, Y/W)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
