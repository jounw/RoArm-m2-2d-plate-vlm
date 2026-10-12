"""RoArm-M2 pick-and-place using an iPad LiDAR (Record3D) as a fixed depth camera.

Same overall cycle as pick_place_fixed_cam.py, but instead of assuming the
object sits on a flat calibrated floor plane (ground_homography.json, 2D
pixel->floor mapping), this reads the LiDAR's actual depth at the detected
object's pixel and back-projects it to a real 3D point, then maps that
point into the robot's coordinate frame via the rigid transform from
calibrate_lidar_extrinsic.py. That means the true TOP-of-object height is
used automatically -- no error from object thickness like the pure-homography
approach has.

RGB carries most of the "what/where" decision; the LiDAR mainly supplies z:
  1. move to start pose, open gripper
  2. grab one Record3D frame (RGB + depth), detect the target
     (VLM / darkest-blob / manual box) on the RGB image
  3. tighten the detected box to the object's own color mask
     (record3d_camera.refine_box_with_color) -- a loose/rough box no longer
     drags in background
  4. average several LiDAR frames (temporal median denoising,
     record3d_camera.average_frame), then estimate the object's near/top
     surface point: RGB color is used FIRST to select which pixels in the
     box belong to the object (Lab color space), and depth is only used
     afterward to pick the near/top surface among those already
     color-selected pixels (record3d_camera.estimate_object_surface_point,
     color_weight=True). Falls back to a depth-only percentile pass, and
     finally to a single median-depth read at the box center, if the color
     mask is too small to trust. Back-projects to a camera-frame 3D point,
     maps to the robot frame with the rigid transform from
     calibrate_lidar_extrinsic.py (lidar_extrinsic.json; 9-point
     calibration recommended for a workspace spread out over
     near/far/left/right/low/high).
  5. move to that (x, y, z + gripper clearance), close the gripper
  6. lift, carry to place location (taught pose OR click-to-place), open, lift
  7. return to start pose

Run: python3 pick_place_lidar.py --detector manual --place-mode click --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from darkest_detector import find_darkest_box
from record3d_camera import (
    Record3DCamera,
    average_frame,
    displayed_to_original_xy,
    estimate_object_surface_point,
    median_depth_around,
    pixel_depth_to_camera_point,
    refine_box_with_color,
    rotate_for_display,
)
from roarm_driver import Pose, RoArm
from vision import locate

FRAME_PATH = "vision_frame_lidar.jpg"
WINDOW_NAME = "RoArm LiDAR Pick-Place"
DEPTH_WINDOW_NAME = "RoArm LiDAR Depth"

_show_depth_window = True


def depth_to_color(depth_m: np.ndarray) -> np.ndarray:
    """Colorize a depth map (meters) for display: near = warm, far = cool,
    invalid (0/inf) pixels shown black."""
    valid_mask = (depth_m > 0.05) & (depth_m < 5.0) & np.isfinite(depth_m)
    vis = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid_mask.any():
        d = depth_m.copy()
        lo, hi = d[valid_mask].min(), d[valid_mask].max()
        span = max(hi - lo, 1e-6)
        vis[valid_mask] = np.clip(255 - (d[valid_mask] - lo) / span * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[~valid_mask] = (0, 0, 0)
    return color


def _tick_depth_window(frame, rotate: str) -> None:
    if not _show_depth_window:
        return
    depth_color = depth_to_color(frame.depth_m)
    cv2.imshow(DEPTH_WINDOW_NAME, rotate_for_display(depth_color, rotate))


def load_start_pose(path: str) -> Pose:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    p = data["base_pose"]
    return Pose(x=float(p["x"]), y=float(p["y"]), z=float(p["z"]), t=float(p["t"]))


def load_place_pose(path: str) -> Pose | None:
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    p = data.get("base_pose", data)
    return Pose(x=float(p["x"]), y=float(p["y"]), z=float(p["z"]), t=float(p["t"]))


def load_extrinsic(path: str) -> tuple[np.ndarray, np.ndarray]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found -- run calibrate_lidar_extrinsic.py first")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    R = np.array(data["rotation"], dtype=np.float64)
    t = np.array(data["translation"], dtype=np.float64)
    return R, t


def camera_point_to_robot(R: np.ndarray, t: np.ndarray, camera_point_mm: np.ndarray) -> np.ndarray:
    return R @ camera_point_mm + t


def normalized_box_to_pixel(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h)


def capture_rgb_for_vlm(rgb: np.ndarray) -> None:
    cv2.imwrite(FRAME_PATH, rgb)


def select_roi_manually(cam: Record3DCamera, first_rgb: np.ndarray, rotate: str) -> tuple[int, int, int, int] | None:
    """Freeze on a live (rotated-for-display) frame and let the user drag a
    box. Returns the box in ORIGINAL (unrotated) frame pixel coordinates."""
    print("  라이브 화면입니다. SPACE/Enter로 화면을 잡은 뒤 물체에 박스를 드래그하세요 (취소: c)")
    orig_frame = first_rgb
    while True:
        f = cam.get_frame()
        orig_frame = f.rgb
        disp = rotate_for_display(orig_frame, rotate).copy()
        cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
        cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        cv2.imshow(WINDOW_NAME, disp)
        _tick_depth_window(f, rotate)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            break
        if key == ord("c"):
            return None
    disp_frame = rotate_for_display(orig_frame, rotate)
    dx, dy, dw, dh = (int(v) for v in cv2.selectROI(WINDOW_NAME, disp_frame, fromCenter=False, showCrosshair=True))
    if dw <= 0 or dh <= 0:
        return None
    orig_h, orig_w = orig_frame.shape[:2]
    # map all 4 corners of the display-space box back to original coordinates
    # and take the bounding box (robust regardless of rotation direction)
    corners = [(dx, dy), (dx + dw, dy), (dx, dy + dh), (dx + dw, dy + dh)]
    mapped = [displayed_to_original_xy(cx, cy, orig_w, orig_h, rotate) for cx, cy in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    x, y, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x, y, x2 - x, y2 - y


def detect_target_pixel_box(
    cam: Record3DCamera, rgb: np.ndarray, target: str | None, detector: str, min_area_frac: float, rotate: str
) -> tuple[int, int, int, int] | None:
    """Returns a pixel (x1, y1, x2, y2) box on the ORIGINAL rgb frame, or None if not found."""
    if detector == "manual":
        roi = select_roi_manually(cam, rgb, rotate)
        if roi is None:
            return None
        x, y, w, h = roi
        return x, y, x + w, y + h
    if detector == "darkest":
        box = find_darkest_box(rgb, min_area_frac=min_area_frac)
        if box is None:
            return None
        x, y, w, h = box
        return x, y, x + w, y + h
    # vlm
    capture_rgb_for_vlm(rgb)
    det = locate(FRAME_PATH, target)
    print(f"  VLM: {det.answer!r}")
    if det.box is None:
        return None
    return normalized_box_to_pixel(det.box, rgb.shape)


def draw_label(frame, org: tuple[int, int], text: str, color: tuple[int, int, int] = (255, 80, 80)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def show_preview(rgb, box, robot_xyz: np.ndarray | None, text: str, rotate: str, depth_m: np.ndarray | None = None) -> None:
    """rgb and box are in ORIGINAL frame coordinates. Drawing happens on a
    copy of the original frame (so box coordinates need no remapping), then
    the whole annotated image is rotated for display."""
    annotated = rgb.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 80, 80), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 5, (255, 80, 80), -1)
        if robot_xyz is not None:
            x, y, z = robot_xyz
            draw_label(annotated, (cx + 10, cy - 10), f"({x:.0f},{y:.0f},{z:.0f})mm")
    disp = rotate_for_display(annotated, rotate)
    cv2.putText(disp, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
    cv2.putText(disp, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    cv2.imshow(WINDOW_NAME, disp)
    if depth_m is not None and _show_depth_window:
        cv2.imshow(DEPTH_WINDOW_NAME, rotate_for_display(depth_to_color(depth_m), rotate))
    cv2.waitKey(1)


def show_live(cam: Record3DCamera, seconds: float, rotate: str) -> None:
    end = time.time() + seconds
    while time.time() < end:
        frame = cam.get_frame()
        cv2.imshow(WINDOW_NAME, rotate_for_display(frame.rgb, rotate))
        _tick_depth_window(frame, rotate)
        cv2.waitKey(30)


def wait_gripper_settled(arm: RoArm, cam: Record3DCamera, rotate: str, timeout: float = 2.5) -> None:
    deadline = time.time() + timeout
    last_t = None
    while time.time() < deadline:
        show_live(cam, 0.15, rotate)
        fb = arm.feedback()
        if fb is None:
            continue
        if last_t is not None and abs(fb.t - last_t) < 0.01:
            return
        last_t = fb.t


def pick(arm: RoArm, cam: Record3DCamera, carry_z: float, grip_settle_sec: float, grip_close_rad: float, rotate: str) -> None:
    arm.set_gripper(grip_close_rad)
    wait_gripper_settled(arm, cam, rotate, timeout=3.0)
    show_live(cam, grip_settle_sec, rotate)
    pose = arm.feedback()
    pose.z = max(pose.z, carry_z)
    arm.move_to(pose)
    show_live(cam, 1.0, rotate)


def select_point_click(cam: Record3DCamera, prompt_text: str, R: np.ndarray, t: np.ndarray, rotate: str) -> tuple[int, int, float] | None:
    """Freeze on the live (rotated-for-display) feed, let the user click a
    drop point, show its live robot-frame (x,y,z) using the current depth
    reading. Returns (px, py, depth_m) in ORIGINAL frame coordinates, or
    None if cancelled."""
    clicked_disp: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_disp.clear()
            clicked_disp.append((x, y))

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print(f"  {prompt_text} (클릭 후 c로 확정, 취소: q)")
    try:
        while True:
            frame = cam.get_frame()
            orig_h, orig_w = frame.rgb.shape[:2]
            disp = rotate_for_display(frame.rgb, rotate).copy()
            if clicked_disp:
                dx, dy = clicked_disp[0]
                px, py = displayed_to_original_xy(dx, dy, orig_w, orig_h, rotate)
                cv2.circle(disp, (dx, dy), 8, (0, 220, 0), -1)
                d = median_depth_around(frame.depth_m, px, py)
                if d is not None:
                    cam_pt = pixel_depth_to_camera_point(px, py, d, frame.intrinsics) * 1000.0
                    robot_pt = camera_point_to_robot(R, t, cam_pt)
                    draw_label(disp, (dx + 12, dy - 12),
                               f"({robot_pt[0]:.0f},{robot_pt[1]:.0f},{robot_pt[2]:.0f})mm", color=(0, 220, 0))
                else:
                    draw_label(disp, (dx + 12, dy - 12), "depth invalid", color=(0, 0, 220))
            cv2.putText(disp, f"{prompt_text} (click, c: confirm, q: cancel)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
            cv2.putText(disp, f"{prompt_text} (click, c: confirm, q: cancel)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
            cv2.imshow(WINDOW_NAME, disp)
            _tick_depth_window(frame, rotate)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("c") and clicked_disp:
                dx, dy = clicked_disp[0]
                px, py = displayed_to_original_xy(dx, dy, orig_w, orig_h, rotate)
                d = median_depth_around(frame.depth_m, px, py)
                if d is None:
                    print("  이 지점의 깊이값이 유효하지 않습니다. 다시 클릭하세요.")
                    clicked_disp.clear()
                    continue
                return px, py, d
    finally:
        cv2.setMouseCallback(WINDOW_NAME, lambda *a: None)


def pick_place_pose_from_click(
    cam: Record3DCamera, R: np.ndarray, t: np.ndarray, place_z_override: float | None, rotate: str, z_lift_mm: float = 0.0
) -> Pose | None:
    result = select_point_click(cam, "놓을 위치를 클릭하세요", R, t, rotate)
    if result is None:
        return None
    px, py, depth_m = result
    frame = cam.get_frame()
    cam_pt = pixel_depth_to_camera_point(px, py, depth_m, frame.intrinsics) * 1000.0
    robot_pt = camera_point_to_robot(R, t, cam_pt)
    z = place_z_override if place_z_override is not None else robot_pt[2] + z_lift_mm
    print(f"  클릭 픽셀=({px},{py}) depth={depth_m:.3f}m -> robot=({robot_pt[0]:.1f},{robot_pt[1]:.1f},{z:.1f}) (표면 z={robot_pt[2]:.1f} + {z_lift_mm:.1f}mm)")
    return Pose(x=robot_pt[0], y=robot_pt[1], z=z, t=RoArm.GRIPPER_OPEN)


def place_at(arm: RoArm, cam: Record3DCamera, place_pose: Pose, carry_z: float, grip_close_rad: float, rotate: str) -> None:
    high_z = max(place_pose.z, carry_z)
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, grip_close_rad))
    show_live(cam, 1.3, rotate)
    arm.move_to(Pose(place_pose.x, place_pose.y, place_pose.z, grip_close_rad))
    show_live(cam, 1.0, rotate)
    arm.open_gripper()
    wait_gripper_settled(arm, cam, rotate, timeout=2.5)
    show_live(cam, 0.3, rotate)
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, RoArm.GRIPPER_OPEN))
    show_live(cam, 0.8, rotate)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RoArm-M2 pick-and-place using an iPad LiDAR (Record3D) depth camera")
    p.add_argument("--target", default=None, help="natural language description of the object (required for --detector vlm)")
    p.add_argument("--detector", choices=("vlm", "darkest", "manual"), default="manual")
    p.add_argument("--vlm-backend", choices=("gemini", "openai"), default=None)
    p.add_argument("--min-area-frac", type=float, default=0.01)
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "/dev/cu.usbserial-120"))
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--device-index", type=int, default=0, help="Record3D device index, if multiple iOS devices are connected")
    p.add_argument("--start-pose", default="roarm_start_pose.json")
    p.add_argument(
        "--place-mode",
        choices=("taught", "click"),
        default="taught",
        help="'taught' uses --place-pose/place_pose.json (fixed spot); "
             "'click' picks up, holds in the air, and waits for you to click the drop point on the live feed each cycle",
    )
    p.add_argument("--place-pose", default="place_pose.json")
    p.add_argument("--place-x", type=float, default=None)
    p.add_argument("--place-y", type=float, default=None)
    p.add_argument("--place-z", type=float, default=None)
    p.add_argument("--click-place-z", type=float, default=None, help="override z when placing by click (default: use the clicked surface's own depth-derived z)")
    p.add_argument("--click-place-z-lift-mm", type=float, default=10.0, help="raise the click-placed z by this much (mm) above the clicked surface's own depth-derived z, so the object is set down slightly higher (default 10mm = 1cm)")
    p.add_argument("--extrinsic", default="lidar_extrinsic.json", help="output of calibrate_lidar_extrinsic.py")
    p.add_argument("--show-depth", dest="show_depth", action="store_true", default=True, help="also show a live colorized depth window from the Record3D LiDAR (on by default)")
    p.add_argument("--no-show-depth", dest="show_depth", action="store_false")
    p.add_argument(
        "--rotate",
        choices=("none", "left", "right", "180"),
        default="left",
        help="rotate the preview window (display only, clicks map back correctly): "
             "'left' = 90deg counter-clockwise (default, matches iPad held upright), 'right' = 90deg clockwise, '180' = upside down",
    )
    p.add_argument("--grip-clearance-mm", type=float, default=5.0, help="stop the gripper this far above the detected object's top surface")
    p.add_argument("--depth-average-frames", type=int, default=5, help="number of LiDAR frames to median-average for a denoised depth reading (1 = disable)")
    p.add_argument("--color-tolerance", type=float, default=28.0, help="Lab color distance threshold used by RGB to select the object's own pixels (both for box refinement and surface estimation); lower = stricter")
    p.add_argument("--no-color-box-refine", dest="color_box_refine", action="store_false", default=True, help="disable tightening the detected box to the object's RGB color mask before reading depth")
    p.add_argument("--approach-speed", type=float, default=0.5, help="arm speed (0-1) for the move to the target")
    p.add_argument("--carry-z", type=float, default=None, help="absolute z to travel at while carrying (default: start pose height)")
    p.add_argument("--grip-settle-sec", type=float, default=0.5)
    p.add_argument("--grip-close-rad", type=float, default=3.25)
    p.add_argument("--max-cycles", type=int, default=0, help="0 = run until Ctrl+C or no object found")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.vlm_backend:
        os.environ["VLM_BACKEND"] = args.vlm_backend

    R, t = load_extrinsic(args.extrinsic)

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)
    start_pose = load_start_pose(args.start_pose)

    place_pose = None
    if args.place_mode == "taught":
        place_pose = load_place_pose(args.place_pose)
        if place_pose is not None:
            print(f"place pose loaded: x={place_pose.x:.1f} y={place_pose.y:.1f} z={place_pose.z:.1f}")
        elif any(v is not None for v in (args.place_x, args.place_y, args.place_z)):
            place_pose = Pose(
                x=args.place_x if args.place_x is not None else start_pose.x,
                y=args.place_y if args.place_y is not None else start_pose.y + 100.0,
                z=args.place_z if args.place_z is not None else start_pose.z,
                t=RoArm.GRIPPER_OPEN,
            )
        else:
            print("놓을 위치가 없습니다 (teach_place.py 실행 권장); 한 번만 집고 시작 자세에서 들고 있습니다.")
    else:
        print("place-mode=click: 매 사이클 물체를 든 채 공중에서 대기하다가, 화면 클릭으로 놓을 위치를 지정합니다.")

    carry_z = args.carry_z if args.carry_z is not None else start_pose.z

    print("Record3D 카메라에 연결 중... (iPad에서 Record3D 앱 -> USB Streaming mode 켜져 있어야 함)")
    cam = Record3DCamera(device_index=args.device_index)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    global _show_depth_window
    _show_depth_window = args.show_depth
    if _show_depth_window:
        cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("라이다 depth 미리보기 창을 띄웠습니다.")

    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"===== cycle {cycle} =====")
            print("moving to start pose")
            arm.move_to(start_pose)
            show_live(cam, 1.0, args.rotate)
            arm.open_gripper()
            wait_gripper_settled(arm, cam, args.rotate, timeout=2.0)

            frame = cam.get_frame()
            box = detect_target_pixel_box(cam, frame.rgb, args.target, args.detector, args.min_area_frac, args.rotate)
            if box is None:
                show_preview(frame.rgb, None, None, "no target found", args.rotate, depth_m=frame.depth_m)
                if cycle == 1:
                    print("failed to locate target, aborting")
                    return 1
                print("no more objects found; stopping loop")
                break

            # RGB-first refinement: tighten the detected box to the object's
            # own color mask before touching depth at all, so a loose/rough
            # box (VLM, darkest-blob, or a hand-drawn manual box) doesn't
            # drag in background pixels
            if args.color_box_refine:
                refined_box = refine_box_with_color(frame.rgb, box, color_tolerance=args.color_tolerance)
                if refined_box != box:
                    print(f"  RGB 색상으로 박스 정교화: {box} -> {refined_box}")
                box = refined_box

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # denoise depth over several frames, then estimate the object's
            # near/top surface, using RGB color as the PRIMARY filter for
            # which pixels belong to the object (depth only picks the near
            # surface among the color-selected pixels)
            denoised = average_frame(cam, n=args.depth_average_frames)
            surface_pt_m = estimate_object_surface_point(
                denoised.depth_m, box, denoised.intrinsics, rgb=denoised.rgb, color_tolerance=args.color_tolerance
            )
            if surface_pt_m is not None:
                cam_pt_mm = surface_pt_m * 1000.0
                object_depth_m = float(surface_pt_m[2])
            else:
                object_depth_m = median_depth_around(denoised.depth_m, cx, cy)
                if object_depth_m is None:
                    print("  물체 위치의 깊이값이 유효하지 않습니다 (반사/거리 문제). 이 사이클을 건너뜁니다.")
                    if cycle == 1:
                        return 1
                    break
                cam_pt_mm = pixel_depth_to_camera_point(cx, cy, object_depth_m, denoised.intrinsics) * 1000.0

            robot_pt = camera_point_to_robot(R, t, cam_pt_mm)
            wx, wy, wz = robot_pt
            grab_z = wz + args.grip_clearance_mm
            show_preview(frame.rgb, box, robot_pt, f"target=({wx:.0f},{wy:.0f},{wz:.0f})mm depth={object_depth_m:.3f}m", args.rotate, depth_m=frame.depth_m)
            print(f"target pixel center=({cx},{cy}) depth={object_depth_m:.3f}m -> robot (x,y,z)=({wx:.1f},{wy:.1f},{wz:.1f}), grab_z={grab_z:.1f}")

            approach_pose = Pose(x=wx, y=wy, z=max(grab_z, carry_z), t=RoArm.GRIPPER_OPEN)
            print("moving above target")
            arm.move_to(approach_pose, speed=args.approach_speed)
            show_live(cam, 1.2, args.rotate)

            print("descending to grab height")
            arm.move_to(Pose(wx, wy, grab_z, RoArm.GRIPPER_OPEN), speed=args.approach_speed)
            show_live(cam, 1.0, args.rotate)

            print("picking")
            pick(arm, cam, carry_z, args.grip_settle_sec, args.grip_close_rad, args.rotate)

            if args.place_mode == "click":
                print("공중에서 대기 중: 놓을 위치를 화면에서 클릭하세요")
                click_place_pose = pick_place_pose_from_click(cam, R, t, args.click_place_z, args.rotate, args.click_place_z_lift_mm)
                if click_place_pose is None:
                    print("취소됨: 물체를 든 채 시작 자세로 복귀합니다 (그립 유지)")
                    arm.move_to(Pose(start_pose.x, start_pose.y, carry_z, args.grip_close_rad))
                    show_live(cam, 1.5, args.rotate)
                    arm.close_gripper()
                    show_live(cam, 0.3, args.rotate)
                    break
                print(f"placing at clicked point: x={click_place_pose.x:.1f} y={click_place_pose.y:.1f} z={click_place_pose.z:.1f}")
                place_at(arm, cam, click_place_pose, carry_z, args.grip_close_rad, args.rotate)
                print("returning to start pose (gripper open)")
                arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
                show_live(cam, 1.0, args.rotate)
            elif place_pose is None:
                print("returning to start pose, holding the object (no place pose -> single cycle)")
                arm.move_to(start_pose)
                show_live(cam, 1.5, args.rotate)
                arm.close_gripper()
                show_live(cam, 0.3, args.rotate)
                break
            else:
                print("placing at taught pose")
                place_at(arm, cam, place_pose, carry_z, args.grip_close_rad, args.rotate)
                print("returning to start pose (gripper open)")
                arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
                show_live(cam, 1.0, args.rotate)

            if args.max_cycles and cycle >= args.max_cycles:
                print(f"reached max cycles ({args.max_cycles}); stopping")
                break
    except KeyboardInterrupt:
        print("\ninterrupted, stopping")
    finally:
        cam.close()
        cv2.destroyWindow(WINDOW_NAME)
        if _show_depth_window:
            cv2.destroyWindow(DEPTH_WINDOW_NAME)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
