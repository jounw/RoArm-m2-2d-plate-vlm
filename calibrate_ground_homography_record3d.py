"""Calibrate a ground-plane homography using the Record3D (iPad) RGB stream
as the fixed camera, instead of a plain cv2.VideoCapture webcam.

Same procedure/output format as calibrate_ground_homography.py, but reads
frames via Record3DCamera and only ever uses frame.rgb (depth is never
touched) -- this produces a ground_homography.json compatible with
pick_place_record3d_rgb.py.

IMPORTANT: pixel_points are saved in the Record3D ORIGINAL (unrotated) frame
coordinate system, using the same displayed_to_original_xy mapping used
elsewhere in this project, so run this with the same --rotate value you
intend to use with pick_place_record3d_rgb.py (default "left" everywhere).

Procedure:
  1. Mark N (>=4) points on the floor within the iPad's view.
  2. Live feed shown rotated for viewing; click each marked point (in any
     order), confirm with 'c'.
  3. Torque off, hand-move the gripper to that same physical point, Enter to
     record the arm's real (x, y).
  4. Repeat for all N points, then solve the homography (4 points: exact
     cv2.getPerspectiveTransform; 5+: least-squares cv2.findHomography) and
     save it, in the same format as calibrate_ground_homography.py.

Run: python3 calibrate_ground_homography_record3d.py --serial-port /dev/cu.usbserial-120 --num-points 9
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from record3d_camera import Record3DCamera, displayed_to_original_xy, rotate_for_display
from roarm_driver import RoArm

OUT_PATH = "ground_homography.json"
NUM_POINTS = 9
WINDOW_NAME = "Ground calibration (Record3D RGB) - click a marked point, then press c"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate pixel->floor(x,y) homography using the Record3D RGB stream.")
    parser.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    parser.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--device-index", type=int, default=0, help="Record3D device index, if multiple iOS devices are connected")
    parser.add_argument("--num-points", type=int, default=NUM_POINTS, help="how many floor points you marked (>=4; 9 recommended, spread across the whole floor area)")
    parser.add_argument(
        "--rotate",
        choices=("none", "left", "right", "180"),
        default="left",
        help="rotate the preview window (display only, clicks map back correctly to the saved pixel_points): "
             "'left' = 90deg counter-clockwise (default, matches iPad held upright), 'right' = 90deg clockwise, '180' = upside down",
    )
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    if args.num_points < 4:
        print("호모그래피 계산에는 최소 4개의 점이 필요합니다.")
        return 1

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)

    print("Record3D 카메라에 연결 중... (iPad에서 Record3D 앱 -> USB Streaming mode 켜져 있어야 함, RGB만 사용합니다)")
    cam = Record3DCamera(device_index=args.device_index)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    clicked_disp: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_disp.clear()
            clicked_disp.append((x, y))

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    pixel_points: list[tuple[int, int]] = []
    world_points: list[tuple[float, float]] = []

    print(f"바닥에 표시해둔 점 {args.num_points}개를 하나씩 처리합니다.")
    if args.num_points >= 8:
        print("  (권장: 바닥 작업 영역 전체에 고르게 흩어서 찍으세요)")
    print("각 점마다: 1) 화면에서 그 점을 클릭  2) 'c'로 확정  3) 팔을 손으로 그 점 위로 옮기고 Enter\n")

    try:
        for i in range(args.num_points):
            print(f"--- 점 {i + 1}/{args.num_points} ---")
            clicked_disp.clear()
            orig_frame = None
            while True:
                f = cam.get_frame()
                orig_frame = f.rgb
                orig_h, orig_w = orig_frame.shape[:2]
                disp = rotate_for_display(orig_frame, args.rotate).copy()
                if clicked_disp:
                    cv2.circle(disp, clicked_disp[0], 7, (0, 220, 0), -1)
                cv2.putText(
                    disp, f"point {i + 1}/{args.num_points}: click it, then press c (q: cancel)",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                )
                cv2.putText(
                    disp, f"point {i + 1}/{args.num_points}: click it, then press c (q: cancel)",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                )
                cv2.imshow(WINDOW_NAME, disp)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    print("취소합니다.")
                    return 1
                if key == ord("c") and clicked_disp:
                    break
            dx, dy = clicked_disp[0]
            px, py = displayed_to_original_xy(dx, dy, orig_w, orig_h, args.rotate)
            print(f"  원본 화면 좌표: ({px}, {py})")

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
        cam.close()
        cv2.destroyWindow(WINDOW_NAME)

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
        "camera_index": "record3d",
        "rotate": args.rotate,
        "homography": H.tolist(),  # 3x3: pixel(x,y,1) -> world(x,y,w), then divide by w -- ORIGINAL (unrotated) frame coords
        "pixel_points": pixel_points,
        "world_points": world_points,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장됨: {args.out}")
    print("사용법: 픽셀 (px, py, ORIGINAL 프레임 좌표) -> [X, Y, W] = H @ [px, py, 1]; 실제 좌표 = (X/W, Y/W)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
