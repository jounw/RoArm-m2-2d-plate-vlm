"""Calibrate the rigid transform from the Record3D (iPad LiDAR) camera's
coordinate frame to the RoArm's base coordinate frame.

Unlike calibrate_ground_homography.py (a 2D pixel->floor-plane mapping,
only valid for objects with ~zero height sitting exactly on the calibrated
floor), this uses the LiDAR's actual per-pixel depth, so it works in full
3D: any pixel + its depth reading converts to a real (x, y, z) in the
robot's coordinate frame, regardless of how tall the object is.

Procedure, for each of N >= 4 points:
  1. Live RGB feed from the iPad; click a point on a visible flat marker/
     surface feature, confirm with 'c'. Depth is read (median over a small
     window for robustness) and back-projected to a 3D point in the
     camera's own coordinate frame.
  2. Torque is turned off; move the gripper by hand to that exact physical
     point; press Enter to record the arm's real (x, y, z) there.
  3. Repeat for all N points.

With >= 3 non-collinear correspondences, solves for the best-fit rotation
+ translation (Kabsch/Umeyama, no scaling since both sides are already in
consistent physical units -- camera points are converted mm on read) that
maps camera-frame points to robot-frame points, and saves it.

Run: python3 calibrate_lidar_extrinsic.py --serial-port /dev/cu.usbserial-120 --num-points 9

9 points (default) spread across the whole workspace (near/far, left/right,
low/high) give the least-squares Kabsch fit more redundancy than the
minimum of 3-4, averaging out per-point click/depth noise and letting the
reported residuals_mm reveal whether error is uniform or concentrated in
one region (e.g. near the edge of the arm's reach).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from record3d_camera import (
    Record3DCamera,
    displayed_to_original_xy,
    median_depth_around,
    pixel_depth_to_camera_point,
    rotate_for_display,
)
from roarm_driver import RoArm

OUT_PATH = "lidar_extrinsic.json"
WINDOW_NAME = "LiDAR Extrinsic Calibration"


def kabsch_umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-fit rigid transform (no scaling) mapping src points to dst points:
    dst ~= R @ src + t. src, dst: Nx3 arrays, N >= 3, not all collinear.
    Returns (R (3x3), t (3,))."""
    assert src.shape == dst.shape and src.shape[0] >= 3

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def select_point_on_frame(cam: Record3DCamera, prompt_text: str, rotate: str = "left"):
    """Show a live RGB feed (rotated for display), let the user click one
    point, confirm with 'c'. Returns (px, py, depth_frame, intrinsics) in
    ORIGINAL frame coordinates, or None if cancelled."""
    clicked_disp: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_disp.clear()
            clicked_disp.append((x, y))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print(f"  {prompt_text} (클릭 후 c로 확정, 취소: q)")

    last_frame = None
    try:
        while True:
            frame = cam.get_frame()
            last_frame = frame
            orig_h, orig_w = frame.rgb.shape[:2]
            disp = rotate_for_display(frame.rgb, rotate).copy()
            if clicked_disp:
                dx, dy = clicked_disp[0]
                px, py = displayed_to_original_xy(dx, dy, orig_w, orig_h, rotate)
                cv2.circle(disp, (dx, dy), 8, (0, 220, 0), -1)
                d = median_depth_around(frame.depth_m, px, py)
                label = f"depth={d:.3f}m" if d is not None else "depth=invalid"
                cv2.putText(disp, label, (dx + 12, dy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(disp, f"{prompt_text} (click, c: confirm, q: cancel)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
            cv2.putText(disp, f"{prompt_text} (click, c: confirm, q: cancel)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("c") and clicked_disp:
                dx, dy = clicked_disp[0]
                px, py = displayed_to_original_xy(dx, dy, orig_w, orig_h, rotate)
                depth = median_depth_around(last_frame.depth_m, px, py)
                if depth is None:
                    dh, dw = last_frame.depth_m.shape[:2]
                    raw = None
                    if 0 <= py < dh and 0 <= px < dw:
                        raw = float(last_frame.depth_m[py, px])
                    print(
                        f"  이 지점의 깊이값을 읽지 못했습니다 (반사/거리 문제일 수 있음). "
                        f"클릭 원본좌표=({px},{py}), depth 배열 크기={dw}x{dh}, 해당 픽셀 원값={raw}. "
                        "다른 지점을 클릭하세요."
                    )
                    clicked_disp.clear()
                    continue
                return px, py, depth, last_frame.intrinsics
    finally:
        cv2.setMouseCallback(WINDOW_NAME, lambda *a: None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate iPad LiDAR (Record3D) -> RoArm base rigid transform.")
    parser.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    parser.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--device-index", type=int, default=0, help="Record3D device index, if multiple iOS devices are connected")
    parser.add_argument("--num-points", type=int, default=9, help="how many correspondences to collect (>=3; 9 spread across the whole workspace recommended for a robust fit)")
    parser.add_argument(
        "--rotate",
        choices=("none", "left", "right", "180"),
        default="left",
        help="rotate the preview window (display only, clicks are mapped back correctly): "
             "'left' = 90deg counter-clockwise (default, matches iPad held upright), 'right' = 90deg clockwise, '180' = upside down",
    )
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    if args.num_points < 3:
        print("최소 3개 지점이 필요합니다 (4개 이상 권장).")
        return 1

    print("Record3D 카메라에 연결 중... (iPad에서 Record3D 앱 -> USB Streaming mode 켜져 있어야 함)")
    cam = Record3DCamera(device_index=args.device_index)

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)

    camera_points: list[list[float]] = []
    robot_points: list[list[float]] = []

    print(f"\n지점 {args.num_points}개를 하나씩 처리합니다.")
    print("각 지점마다: 1) 라이브 화면에서 그 지점을 클릭 후 c  2) 팔을 손으로 같은 물리적 지점으로 옮기고 Enter")
    if args.num_points >= 8:
        print("  (권장: 작업 공간 전체에 고르게 -- 가까이/멀리, 좌/우, 낮게/높게 섞어서 찍으세요)\n")
    else:
        print()

    try:
        for i in range(args.num_points):
            print(f"--- 지점 {i + 1}/{args.num_points} ---")
            result = select_point_on_frame(cam, f"지점 {i + 1}/{args.num_points}: 클릭하세요", rotate=args.rotate)
            if result is None:
                print("취소합니다.")
                return 1
            px, py, depth_m, intrinsics = result
            cam_point_m = pixel_depth_to_camera_point(px, py, depth_m, intrinsics)
            cam_point_mm = cam_point_m * 1000.0
            print(f"  화면 좌표=({px},{py}), depth={depth_m:.3f}m -> camera-frame point={cam_point_mm.round(1)}mm")

            print("  토크를 끕니다. 그리퍼 끝을 방금 클릭한 지점 위로 옮기세요.")
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
            print(f"  측정된 로봇 좌표: x={pose.x:.1f}, y={pose.y:.1f}, z={pose.z:.1f}")

            camera_points.append(cam_point_mm.tolist())
            robot_points.append([pose.x, pose.y, pose.z])

        print("토크를 다시 켭니다.")
        arm.torque(True)
        time.sleep(0.3)
    finally:
        cv2.destroyWindow(WINDOW_NAME)
        cam.close()

    src = np.array(camera_points, dtype=np.float64)
    dst = np.array(robot_points, dtype=np.float64)
    R, t = kabsch_umeyama(src, dst)

    # report fit residuals so the user can judge quality
    predicted = (R @ src.T).T + t
    residuals = np.linalg.norm(predicted - dst, axis=1)
    print("\n적합 결과 (예측 vs 실제, mm 오차):")
    for i, r in enumerate(residuals):
        print(f"  지점 {i + 1}: 오차 {r:.2f}mm")
    print(f"  평균 오차: {residuals.mean():.2f}mm, 최대 오차: {residuals.max():.2f}mm")

    data = {
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "camera_points_mm": camera_points,
        "robot_points_mm": robot_points,
        "residuals_mm": residuals.tolist(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장됨: {args.out}")
    print("사용법: robot_point_mm = R @ camera_point_mm + t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
