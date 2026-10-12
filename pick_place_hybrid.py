"""RoArm-M2 pick-and-place driven entirely by the iPad LiDAR (Record3D).

The eye-in-hand webcam (camera index 0, mounted on the gripper) is shown
live purely as a REFERENCE window -- so you can see what the gripper sees
while it moves -- but it plays no role in detection, alignment, or descent.
All positioning (finding the target, aligning to it, descending to grab
height, and click-to-place) comes from the LiDAR's real depth-derived 3D
coordinates and the calibrated LiDAR->robot extrinsic transform.

Accuracy comes from the LiDAR pipeline itself (RGB is used for more than
just detection/display), not from a second camera:
  - RGB detection: the target's pixel location comes from the RGB frame
    (VLM box, darkest-blob, or manual drag) -- this is where "what/where"
    is decided.
  - temporal denoising: once a target/click point is known, several LiDAR
    frames are averaged (per-pixel median) before computing the 3D point,
    smoothing out frame-to-frame depth jitter and dropouts
    (record3d_camera.average_frame)
  - spatial surface estimation: instead of trusting one center pixel, every
    valid depth sample inside the detected box is used to find the
    object's near (top) surface and its centroid, robust to loose boxes
    that include some background and to isolated bad pixels
    (record3d_camera.estimate_object_surface_point)
  - RGB color refinement: the surface estimate above is further refined
    using the RGB color of the near-surface pixels -- any pixel inside the
    box whose color doesn't match the object's own (in Lab color space) is
    dropped before computing the final 3D centroid, even if its depth
    happened to match. This catches same-depth background sneaking into a
    loose or slightly-off box, which depth alone can't tell apart.

Loop, once per cycle:
  1. move to start pose, open gripper
  2. LiDAR: detect the target (VLM / darkest-blob / manual box) on a live
     RGB frame, then re-read depth from several averaged frames and
     estimate the object's 3D surface point over the whole box
  3. move straight to that (x, y, z + gripper clearance), close the gripper
  4. lift, carry to place location (taught pose OR click-to-place, same
     averaged-depth treatment), open, lift
  5. return to start pose

(the webcam window keeps streaming throughout every wait/move so you always
have a live look at the gripper, but nothing in the pick logic reads from it)

Run: python3 pick_place_hybrid.py --detector manual --place-mode click --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from darkest_detector import find_darkest_box
from pick_place import (
    WORKSPACE,
    clamp,
    expand_workspace,
    load_workspace_calibration,
    pick as eih_pick,
    place_at as eih_place_at,
    RecordingCapture,
    sleep_showing as eih_sleep_showing,
    wait_gripper_settled as eih_wait_gripper_settled,
)
from record3d_camera import (
    Record3DCamera,
    average_frame,
    displayed_to_original_xy,
    estimate_object_surface_point,
    median_depth_around,
    pixel_depth_to_camera_point,
    rotate_for_display,
)
from roarm_driver import Pose, RoArm
from vision import locate

LIDAR_FRAME_PATH = "vision_frame_hybrid_lidar.jpg"
LIDAR_WINDOW_NAME = "RoArm LiDAR (detection + control)"
DEPTH_WINDOW_NAME = "RoArm LiDAR Depth"

_show_depth_window = True


# ---- LiDAR-side helpers (detection, depth, coordinate math) ------------

def depth_to_color(depth_m: np.ndarray) -> np.ndarray:
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
    cv2.imshow(DEPTH_WINDOW_NAME, rotate_for_display(depth_to_color(frame.depth_m), rotate))


def load_extrinsic(path: str) -> tuple[np.ndarray, np.ndarray]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found -- run calibrate_lidar_extrinsic.py first")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(data["rotation"], dtype=np.float64), np.array(data["translation"], dtype=np.float64)


def camera_point_to_robot(R: np.ndarray, t: np.ndarray, camera_point_mm: np.ndarray) -> np.ndarray:
    return R @ camera_point_mm + t


def normalized_box_to_pixel(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h)


def draw_label(frame, org: tuple[int, int], text: str, color: tuple[int, int, int] = (255, 80, 80)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def select_roi_manually_lidar(cam: Record3DCamera, rotate: str) -> tuple[int, int, int, int] | None:
    """Freeze-and-drag box selection, done on the DEPTH visualization window
    (not the RGB window) -- so what you drag around is literally the LiDAR
    depth reading. RGB keeps streaming in its own window for reference only.
    Returns the box in ORIGINAL (unrotated) frame pixel coordinates."""
    print("  [LiDAR] Depth 화면입니다. SPACE/Enter로 화면을 잡은 뒤 물체에 박스를 드래그하세요 (취소: c)")
    orig_frame = None
    orig_depth_color = None
    while True:
        f = cam.get_frame()
        orig_frame = f.rgb
        orig_depth_color = depth_to_color(f.depth_m)

        # RGB window: plain reference stream
        cv2.imshow(LIDAR_WINDOW_NAME, rotate_for_display(orig_frame, rotate))

        # Depth window: this is where selection actually happens
        disp = rotate_for_display(orig_depth_color, rotate).copy()
        cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
        cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        cv2.imshow(DEPTH_WINDOW_NAME, disp)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            break
        if key == ord("c"):
            return None
    disp_depth_frame = rotate_for_display(orig_depth_color, rotate)
    dx, dy, dw, dh = (int(v) for v in cv2.selectROI(DEPTH_WINDOW_NAME, disp_depth_frame, fromCenter=False, showCrosshair=True))
    if dw <= 0 or dh <= 0:
        return None
    orig_h, orig_w = orig_frame.shape[:2]
    corners = [(dx, dy), (dx + dw, dy), (dx, dy + dh), (dx + dw, dy + dh)]
    mapped = [displayed_to_original_xy(cx, cy, orig_w, orig_h, rotate) for cx, cy in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    x, y, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x, y, x2 - x, y2 - y


def detect_target_lidar(
    cam: Record3DCamera, rgb: np.ndarray, target: str | None, detector: str, min_area_frac: float, rotate: str
) -> tuple[int, int, int, int] | None:
    """Returns a pixel (x1, y1, x2, y2) box on the ORIGINAL LiDAR rgb frame, or None."""
    if detector == "manual":
        roi = select_roi_manually_lidar(cam, rotate)
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
    cv2.imwrite(LIDAR_FRAME_PATH, rgb)
    det = locate(LIDAR_FRAME_PATH, target)
    print(f"  [LiDAR] VLM: {det.answer!r}")
    if det.box is None:
        return None
    return normalized_box_to_pixel(det.box, rgb.shape)


def show_lidar_preview(rgb, box, robot_xyz: np.ndarray | None, text: str, rotate: str, depth_m: np.ndarray | None = None) -> None:
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
    cv2.imshow(LIDAR_WINDOW_NAME, disp)
    if depth_m is not None and _show_depth_window:
        cv2.imshow(DEPTH_WINDOW_NAME, rotate_for_display(depth_to_color(depth_m), rotate))
    cv2.waitKey(1)


def locate_target_with_lidar(
    cam: Record3DCamera, R: np.ndarray, t: np.ndarray, target: str | None, detector: str, min_area_frac: float, rotate: str,
    depth_average_frames: int = 5, color_tolerance: float = 28.0,
) -> np.ndarray | None:
    """One LiDAR look: detect the target (on a live frame), then re-read
    depth from several AVERAGED frames (denoised, dropout-robust) and
    estimate the object's near (top) surface from every valid depth sample
    inside the box -- not just its center pixel. Returns the robot-frame
    (x, y, z) in mm, or None if not found/no valid depth.

    This is the accuracy-focused version: detection still happens on a
    single live frame (needs to react to what's currently in view), but the
    depth/coordinate estimate that actually drives the arm comes from a
    temporally-denoised, spatially-averaged read over the whole box.
    """
    live_frame = cam.get_frame()
    box = detect_target_lidar(cam, live_frame.rgb, target, detector, min_area_frac, rotate)
    if box is None:
        show_lidar_preview(live_frame.rgb, None, None, "[LiDAR] no target found", rotate, depth_m=live_frame.depth_m)
        return None

    # re-grab depth as a temporal average now that we know where to look and
    # the arm/scene should be still -- much less noisy than a single frame
    frame = average_frame(cam, n=depth_average_frames)

    surface_pt_m = estimate_object_surface_point(
        frame.depth_m, box, frame.intrinsics, rgb=frame.rgb, color_tolerance=color_tolerance
    )
    if surface_pt_m is None:
        # fall back to the old center-pixel read if the box has too few valid samples
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        depth_m = median_depth_around(frame.depth_m, cx, cy)
        if depth_m is None:
            show_lidar_preview(frame.rgb, box, None, "[LiDAR] invalid depth at target", rotate, depth_m=frame.depth_m)
            return None
        cam_pt_mm = pixel_depth_to_camera_point(cx, cy, depth_m, frame.intrinsics) * 1000.0
        depth_report = depth_m
    else:
        cam_pt_mm = surface_pt_m * 1000.0
        depth_report = float(surface_pt_m[2])

    robot_pt = camera_point_to_robot(R, t, cam_pt_mm)
    wx, wy, wz = robot_pt
    show_lidar_preview(
        frame.rgb, box, robot_pt, f"[LiDAR] target=({wx:.0f},{wy:.0f},{wz:.0f})mm depth={depth_report:.3f}m", rotate,
        depth_m=frame.depth_m,
    )
    print(f"[LiDAR] box={box} depth={depth_report:.3f}m -> robot (x,y,z)=({wx:.1f},{wy:.1f},{wz:.1f}) (surface-averaged over {depth_average_frames} frames)")
    return robot_pt


def select_point_click_lidar(cam: Record3DCamera, R: np.ndarray, t: np.ndarray, rotate: str) -> tuple[int, int, float] | None:
    """Click-to-place on the LiDAR feed (same UX as pick_place_lidar.py)."""
    clicked_disp: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_disp.clear()
            clicked_disp.append((x, y))

    cv2.setMouseCallback(LIDAR_WINDOW_NAME, on_mouse)
    print("  [LiDAR] 놓을 위치를 클릭하세요 (클릭 후 c로 확정, 취소: q)")
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
            cv2.putText(disp, "click drop point, then c (q: cancel)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
            cv2.putText(disp, "click drop point, then c (q: cancel)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
            cv2.imshow(LIDAR_WINDOW_NAME, disp)
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
        cv2.setMouseCallback(LIDAR_WINDOW_NAME, lambda *a: None)


def pick_place_pose_from_click_lidar(
    cam: Record3DCamera, R: np.ndarray, t: np.ndarray, place_z_override: float | None, rotate: str, z_lift_mm: float,
    depth_average_frames: int = 5,
) -> Pose | None:
    result = select_point_click_lidar(cam, R, t, rotate)
    if result is None:
        return None
    px, py, _live_depth_m = result
    # re-read depth at the confirmed click point from an averaged (denoised)
    # frame, same accuracy treatment as the pick side, instead of trusting
    # the single live frame the click happened to land on
    frame = average_frame(cam, n=depth_average_frames)
    depth_m = median_depth_around(frame.depth_m, px, py)
    if depth_m is None:
        depth_m = _live_depth_m  # fall back to the live reading if averaging lost this pixel
    cam_pt = pixel_depth_to_camera_point(px, py, depth_m, frame.intrinsics) * 1000.0
    robot_pt = camera_point_to_robot(R, t, cam_pt)
    z = place_z_override if place_z_override is not None else robot_pt[2] + z_lift_mm
    print(f"  [LiDAR] 클릭=({px},{py}) depth={depth_m:.3f}m (averaged over {depth_average_frames} frames) -> robot=({robot_pt[0]:.1f},{robot_pt[1]:.1f},{z:.1f})")
    return Pose(x=robot_pt[0], y=robot_pt[1], z=z, t=RoArm.GRIPPER_OPEN)


# ---- main -----------------------------------------------------------

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


def load_floor_z(path: str) -> float | None:
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data["z"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RoArm-M2 pick-and-place driven entirely by the iPad LiDAR; webcam is reference-only")
    p.add_argument("--target", default=None, help="natural language description of the object (required for --detector vlm)")
    p.add_argument("--detector", choices=("vlm", "darkest", "manual"), default="manual", help="detector used on the LiDAR RGB frame")
    p.add_argument("--vlm-backend", choices=("gemini", "openai"), default=None)
    p.add_argument("--min-area-frac", type=float, default=0.01)
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "/dev/cu.usbserial-120"))
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--start-pose", default="roarm_start_pose.json")

    # LiDAR side -- does all detection, alignment, and coordinate math
    p.add_argument("--lidar-device-index", type=int, default=0, help="Record3D device index")
    p.add_argument("--extrinsic", default="lidar_extrinsic.json", help="output of calibrate_lidar_extrinsic.py")
    p.add_argument("--lidar-rotate", choices=("none", "left", "right", "180"), default="left",
                    help="rotate the LiDAR preview window (display only): 'left' = 90deg CCW (default, iPad held upright)")
    p.add_argument("--show-depth", dest="show_depth", action="store_true", default=True)
    p.add_argument("--no-show-depth", dest="show_depth", action="store_false")
    p.add_argument("--approach-speed", type=float, default=0.5, help="arm speed (0-1) for the LiDAR-guided moves")
    p.add_argument("--grip-clearance-mm", type=float, default=5.0, help="stop the gripper this far above the LiDAR-estimated object top")
    p.add_argument("--depth-average-frames", type=int, default=5,
                    help="number of LiDAR frames to average (temporal denoising) when computing the pick/place 3D point; higher = steadier but slower per look (1 disables averaging)")
    p.add_argument("--color-tolerance", type=float, default=28.0,
                    help="RGB(Lab) color distance tolerance used to refine the depth-based surface estimate to the object's own color, excluding same-depth background pixels inside a loose box; lower = stricter")

    # eye-in-hand webcam -- reference display ONLY, not used for detection/positioning
    p.add_argument("--eih-camera-index", type=int, default=0, help="webcam index (mounted on the gripper); shown live for reference only")
    p.add_argument("--record", dest="record", action="store_true", default=True)
    p.add_argument("--no-record", dest="record", action="store_false")

    p.add_argument("--grip-settle-sec", type=float, default=0.5)
    p.add_argument("--grip-close-rad", type=float, default=3.25)
    p.add_argument("--floor-calibration", default="floor_calibration.json")
    p.add_argument("--workspace-calibration", default="workspace_calibration.json")
    p.add_argument("--carry-z", type=float, default=None)

    # place side
    p.add_argument(
        "--place-mode", choices=("taught", "click"), default="taught",
        help="'taught' uses --place-pose/place_pose.json; 'click' holds the object in the air and waits for a LiDAR-feed click each cycle",
    )
    p.add_argument("--place-pose", default="place_pose.json")
    p.add_argument("--place-x", type=float, default=None)
    p.add_argument("--place-y", type=float, default=None)
    p.add_argument("--place-z", type=float, default=None)
    p.add_argument("--click-place-z", type=float, default=None)
    p.add_argument("--click-place-z-lift-mm", type=float, default=10.0)
    p.add_argument("--max-cycles", type=int, default=0)
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
        print("place-mode=click: 매 사이클 물체를 든 채 공중에서 대기하다가, LiDAR 화면 클릭으로 놓을 위치를 지정합니다.")

    floor_z = load_floor_z(args.floor_calibration)
    if floor_z is not None:
        print(f"floor calibration loaded: z={floor_z:.1f} (used only to widen the workspace z lower bound)")

    # set up pick_place.py's module-level WORKSPACE the same way pick_place.main() does
    expand_workspace(start_pose)
    if place_pose is not None:
        expand_workspace(place_pose)
    measured_workspace = load_workspace_calibration(args.workspace_calibration)
    if measured_workspace is not None:
        WORKSPACE["x"] = measured_workspace["x"]
        WORKSPACE["y"] = measured_workspace["y"]
    if floor_z is not None:
        WORKSPACE["z"] = (min(WORKSPACE["z"][0], floor_z - 10.0), WORKSPACE["z"][1])
    print(f"workspace: x={WORKSPACE['x']}, y={WORKSPACE['y']}, z={WORKSPACE['z']}")

    carry_z = clamp(args.carry_z if args.carry_z is not None else start_pose.z, "z")
    print(f"carry height (transit z): {carry_z:.1f}")

    print("Record3D 카메라에 연결 중... (iPad에서 Record3D 앱 -> USB Streaming mode 켜져 있어야 함)")
    lidar_cam = Record3DCamera(device_index=args.lidar_device_index)
    cv2.namedWindow(LIDAR_WINDOW_NAME, cv2.WINDOW_NORMAL)

    global _show_depth_window
    _show_depth_window = args.show_depth
    if args.detector == "manual" and not _show_depth_window:
        print("--detector manual은 Depth 창에서 드래그로 선택합니다; --no-show-depth와 함께 쓸 수 없어 depth 창을 강제로 켭니다.")
        _show_depth_window = True
    if _show_depth_window:
        cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_NORMAL)

    eih_cap = RecordingCapture(args.eih_camera_index) if args.record else cv2.VideoCapture(args.eih_camera_index)
    if not eih_cap.isOpened():
        raise RuntimeError(f"cannot open eye-in-hand camera index {args.eih_camera_index}")
    if args.record and getattr(eih_cap, "path", None):
        print(f"eye-in-hand 카메라 영상 녹화 중: {eih_cap.path}")
    EIH_WINDOW_NAME = "RoArm Eye-in-Hand"
    cv2.namedWindow(EIH_WINDOW_NAME, cv2.WINDOW_NORMAL)
    # pick_place.py's helpers hardcode its own WINDOW_NAME constant; patch it once
    # so the eye-in-hand window title matches what we just created here too.
    import pick_place as _pp
    _pp.WINDOW_NAME = EIH_WINDOW_NAME

    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"===== cycle {cycle} =====")
            print("moving to start pose")
            arm.move_to(start_pose)
            eih_sleep_showing(eih_cap, 1.0)
            arm.open_gripper()
            eih_wait_gripper_settled(arm, eih_cap, timeout=2.0)

            print("[LiDAR] 물체 위치 탐색 중...")
            robot_pt = locate_target_with_lidar(
                lidar_cam, R, t, args.target, args.detector, args.min_area_frac, args.lidar_rotate,
                depth_average_frames=args.depth_average_frames, color_tolerance=args.color_tolerance,
            )
            if robot_pt is None:
                if cycle == 1:
                    print("LiDAR로 물체를 찾지 못했습니다, 중단합니다")
                    return 1
                print("더 이상 물체를 찾지 못함; 종료합니다")
                break

            wx, wy, wz = robot_pt
            grab_z = clamp(wz + args.grip_clearance_mm, "z")
            approach_z = clamp(max(grab_z, carry_z), "z")
            approach_pose = Pose(x=clamp(wx, "x"), y=clamp(wy, "y"), z=approach_z, t=RoArm.GRIPPER_OPEN)
            print(f"[LiDAR->approach] x={approach_pose.x:.1f} y={approach_pose.y:.1f} z={approach_pose.z:.1f}")
            arm.move_to(approach_pose, speed=args.approach_speed)
            eih_sleep_showing(eih_cap, 1.2)

            print(f"[LiDAR->descend] grab_z={grab_z:.1f}")
            arm.move_to(Pose(wx, wy, grab_z, RoArm.GRIPPER_OPEN), speed=args.approach_speed)
            eih_sleep_showing(eih_cap, 1.0)

            print("picking")
            eih_pick(arm, eih_cap, carry_z, args.grip_settle_sec, args.grip_close_rad)

            if args.place_mode == "click":
                print("공중에서 대기 중: LiDAR 화면에서 놓을 위치를 클릭하세요")
                click_place_pose = pick_place_pose_from_click_lidar(
                    lidar_cam, R, t, args.click_place_z, args.lidar_rotate, args.click_place_z_lift_mm,
                    depth_average_frames=args.depth_average_frames,
                )
                if click_place_pose is None:
                    print("취소됨: 물체를 든 채 시작 자세로 복귀합니다 (그립 유지)")
                    arm.move_to(Pose(start_pose.x, start_pose.y, carry_z, args.grip_close_rad))
                    eih_sleep_showing(eih_cap, 1.5)
                    arm.close_gripper()
                    eih_sleep_showing(eih_cap, 0.3)
                    break
                print(f"placing at clicked point: x={click_place_pose.x:.1f} y={click_place_pose.y:.1f} z={click_place_pose.z:.1f}")
                eih_place_at(arm, eih_cap, click_place_pose, carry_z, args.grip_close_rad)
                print("returning to start pose (gripper open)")
                arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
                eih_sleep_showing(eih_cap, 1.0)
            elif place_pose is None:
                print("returning to start pose, holding the object (no place pose -> single cycle)")
                arm.move_to(start_pose)
                eih_sleep_showing(eih_cap, 1.5)
                arm.close_gripper()
                eih_sleep_showing(eih_cap, 0.3)
                break
            else:
                print("placing at taught pose")
                eih_place_at(arm, eih_cap, place_pose, carry_z, args.grip_close_rad)
                print("returning to start pose (gripper open)")
                arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
                eih_sleep_showing(eih_cap, 1.0)

            if args.max_cycles and cycle >= args.max_cycles:
                print(f"reached max cycles ({args.max_cycles}); stopping")
                break
    except KeyboardInterrupt:
        print("\ninterrupted, stopping")
    finally:
        lidar_cam.close()
        cv2.destroyWindow(LIDAR_WINDOW_NAME)
        if _show_depth_window:
            cv2.destroyWindow(DEPTH_WINDOW_NAME)
        eih_cap.release()
        cv2.destroyWindow(EIH_WINDOW_NAME)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
