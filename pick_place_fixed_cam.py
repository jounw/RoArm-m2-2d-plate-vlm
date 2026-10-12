"""RoArm-M2 pick-and-place using a FIXED (non eye-in-hand) overview camera.

Unlike pick_place.py, this does not assume the camera moves with the
gripper. Instead it uses a one-time ground-plane homography (from
calibrate_ground_homography.py) to convert any pixel in the fixed camera's
view directly into a real-world (x, y) on the floor. Loop, once per cycle:

  1. move to start pose, open gripper
  2. capture one frame from the fixed camera, detect the target
     (VLM / darkest-blob / manual box)
  3. map the detected box's center pixel -> real (x, y) via the homography
  4. move straight to that (x, y) at the calibrated floor height (+ clearance),
     close the gripper
  5. lift, carry to the taught place pose, open the gripper, lift
  6. return to start pose

No iterative visual servoing here (no camera-moves-with-gripper feedback
loop to run) -- one look, one geometric solve, one move. Simpler and more
accurate than trying to reuse the eye-in-hand approach math for a camera
that never moves.

Run: python3 pick_place_fixed_cam.py --detector manual --camera-index 1 --serial-port /dev/cu.usbserial-120
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
from roarm_driver import Pose, RoArm
from vision import locate

FRAME_PATH = "vision_frame_fixed.jpg"
WINDOW_NAME = "RoArm Fixed-Camera Pick-Place"
SECONDARY_WINDOW_NAME = "RoArm Secondary Camera"


class SecondaryPreview:
    """Optional second camera shown purely as a live reference window
    (e.g. a wider background view alongside the camera actually used for
    detection/homography). Never touches control logic."""

    def __init__(self, camera_index: int | None):
        self.cap = None
        if camera_index is not None:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                self.cap = cap
                cv2.namedWindow(SECONDARY_WINDOW_NAME, cv2.WINDOW_NORMAL)
                print(f"보조 카메라 index {camera_index} 미리보기 창을 띄웠습니다.")
            else:
                print(f"보조 카메라 index {camera_index}를 열지 못했습니다 (건너뜁니다).")

    def tick(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            cv2.imshow(SECONDARY_WINDOW_NAME, frame)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            cv2.destroyWindow(SECONDARY_WINDOW_NAME)


_secondary_preview: SecondaryPreview | None = None


def _tick_secondary() -> None:
    if _secondary_preview is not None:
        _secondary_preview.tick()


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


def load_floor_z(path: str) -> float:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found -- run calibrate_floor.py first")
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["z"])


def load_homography(path: str) -> tuple[np.ndarray, int]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found -- run calibrate_ground_homography.py first")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    H = np.array(data["homography"], dtype=np.float64)
    return H, int(data.get("camera_index", 1))


def pixel_to_world(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    v = H @ np.array([px, py, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def world_to_pixel(H_inv: np.ndarray, wx: float, wy: float) -> tuple[int, int]:
    v = H_inv @ np.array([wx, wy, 1.0])
    return int(round(v[0] / v[2])), int(round(v[1] / v[2]))


class GroundGridOverlay:
    """Draws the 4 calibrated ground points as a quad, plus an evenly-spaced
    (in real mm) grid inside it -- an AR-style overlay of the calibrated
    working area on the live feed. Built once from ground_homography.json;
    the grid follows perspective because each line is generated in world
    space and projected back to pixels with the inverse homography."""

    def __init__(self, homography_path: str, grid_spacing_mm: float = 50.0):
        self.enabled = False
        self.corners_px: list[tuple[int, int]] = []
        self.grid_lines_px: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if not Path(homography_path).exists():
            return
        data = json.loads(Path(homography_path).read_text(encoding="utf-8"))
        pixel_points = data.get("pixel_points")
        world_points = data.get("world_points")
        H = np.array(data["homography"], dtype=np.float64)
        if not pixel_points or not world_points or len(pixel_points) < 4:
            return
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return

        self.corners_px = [(int(x), int(y)) for x, y in pixel_points]

        xs = [w[0] for w in world_points]
        ys = [w[1] for w in world_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        lines = []
        n_x = max(1, int(round((x_max - x_min) / grid_spacing_mm)))
        n_y = max(1, int(round((y_max - y_min) / grid_spacing_mm)))
        # lines of constant x, spanning y_min..y_max
        for i in range(n_x + 1):
            wx = x_min + (x_max - x_min) * i / n_x
            p1 = world_to_pixel(H_inv, wx, y_min)
            p2 = world_to_pixel(H_inv, wx, y_max)
            lines.append((p1, p2))
        # lines of constant y, spanning x_min..x_max
        for j in range(n_y + 1):
            wy = y_min + (y_max - y_min) * j / n_y
            p1 = world_to_pixel(H_inv, x_min, wy)
            p2 = world_to_pixel(H_inv, x_max, wy)
            lines.append((p1, p2))
        self.grid_lines_px = lines
        self.enabled = True

    def draw(self, frame) -> None:
        if not self.enabled:
            return
        overlay = frame.copy()
        for p1, p2 in self.grid_lines_px:
            cv2.line(overlay, p1, p2, (0, 200, 255), 1, cv2.LINE_AA)
        if len(self.corners_px) >= 4:
            pts = np.array(self.corners_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 220, 0), thickness=2, lineType=cv2.LINE_AA)
            for i, (x, y) in enumerate(self.corners_px):
                cv2.circle(overlay, (x, y), 5, (0, 220, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)


_ground_grid: GroundGridOverlay | None = None


def _draw_grid(frame) -> None:
    if _ground_grid is not None:
        _ground_grid.draw(frame)


def capture_frame(cap: cv2.VideoCapture):
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("failed to read a camera frame")
    cv2.imwrite(FRAME_PATH, frame)
    return frame


def select_roi_manually(cap: cv2.VideoCapture, frame) -> tuple[int, int, int, int] | None:
    print("  라이브 화면입니다. SPACE/Enter로 화면을 잡은 뒤 물체에 박스를 드래그하세요 (취소: c)")
    while True:
        ok, live = cap.read()
        if ok:
            frame = live
            disp = live.copy()
            cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
            cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
            _draw_grid(disp)
            cv2.imshow(WINDOW_NAME, disp)
        _tick_secondary()
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            break
        if key == ord("c"):
            return None
    x, y, w, h = (int(v) for v in cv2.selectROI(WINDOW_NAME, frame, fromCenter=False, showCrosshair=True))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def normalized_box_to_pixel(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h)


def detect_target_pixel_box(
    cap: cv2.VideoCapture, frame, target: str | None, detector: str, min_area_frac: float
) -> tuple[int, int, int, int] | None:
    """Returns a pixel (x1, y1, x2, y2) box, or None if not found."""
    if detector == "manual":
        roi = select_roi_manually(cap, frame)
        if roi is None:
            return None
        x, y, w, h = roi
        return x, y, x + w, y + h
    if detector == "darkest":
        box = find_darkest_box(frame, min_area_frac=min_area_frac)
        if box is None:
            return None
        x, y, w, h = box
        return x, y, x + w, y + h
    # vlm
    det = locate(FRAME_PATH, target)
    print(f"  VLM: {det.answer!r}")
    if det.box is None:
        return None
    return normalized_box_to_pixel(det.box, frame.shape)


def draw_label(frame, org: tuple[int, int], text: str, color: tuple[int, int, int] = (255, 80, 80)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def show_preview(frame, box, world_xy: tuple[float, float] | None, text: str) -> None:
    if box is not None:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 80), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 80, 80), -1)
        if world_xy is not None:
            wx, wy = world_xy
            draw_label(frame, (cx + 10, cy - 10), f"({wx:.0f}, {wy:.0f})mm")
    _draw_grid(frame)
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)


def sleep_showing(cap: cv2.VideoCapture, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        ok, frame = cap.read()
        if ok:
            _draw_grid(frame)
            cv2.imshow(WINDOW_NAME, frame)
        _tick_secondary()
        cv2.waitKey(30)


def wait_gripper_settled(arm: RoArm, cap: cv2.VideoCapture, timeout: float = 2.5) -> None:
    deadline = time.time() + timeout
    last_t = None
    while time.time() < deadline:
        sleep_showing(cap, 0.15)
        fb = arm.feedback()
        if fb is None:
            continue
        if last_t is not None and abs(fb.t - last_t) < 0.01:
            return
        last_t = fb.t


def pick(arm: RoArm, cap: cv2.VideoCapture, carry_z: float, grip_settle_sec: float, grip_close_rad: float) -> None:
    arm.set_gripper(grip_close_rad)
    wait_gripper_settled(arm, cap, timeout=3.0)
    sleep_showing(cap, grip_settle_sec)
    pose = arm.feedback()
    pose.z = max(pose.z, carry_z)
    arm.move_to(pose)
    sleep_showing(cap, 1.0)


def select_point_click(cap: cv2.VideoCapture, prompt_text: str, H: np.ndarray | None = None) -> tuple[int, int] | None:
    """Freeze on a live frame and let the user click one point on it.
    Returns the clicked pixel (x, y), or None if cancelled.
    If H is given, shows the clicked point's real-world (x, y) next to it live."""
    clicked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear()
            clicked.append((x, y))

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print(f"  {prompt_text} (클릭 후 c로 확정, 취소: q)")
    frame = None
    try:
        while True:
            ok, live = cap.read()
            if ok:
                frame = live.copy()
                if clicked:
                    cv2.circle(frame, clicked[0], 8, (0, 220, 0), -1)
                    if H is not None:
                        wx, wy = pixel_to_world(H, clicked[0][0], clicked[0][1])
                        draw_label(frame, (clicked[0][0] + 12, clicked[0][1] - 12), f"({wx:.0f}, {wy:.0f})mm", color=(0, 220, 0))
                cv2.putText(frame, f"{prompt_text}  (click, then c; q: cancel)", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
                cv2.putText(frame, f"{prompt_text}  (click, then c; q: cancel)", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
                _draw_grid(frame)
                cv2.imshow(WINDOW_NAME, frame)
            _tick_secondary()
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("c") and clicked:
                return clicked[0]
    finally:
        cv2.setMouseCallback(WINDOW_NAME, lambda *a: None)


def pick_place_pose_from_click(cap: cv2.VideoCapture, H: np.ndarray, place_z: float) -> Pose | None:
    """Let the user click where to put the (already-held) object down on the
    live feed, and convert that pixel into a place Pose via the homography.
    Returns None if the user cancelled."""
    pt = select_point_click(cap, "놓을 위치를 클릭하세요", H=H)
    if pt is None:
        return None
    px, py = pt
    wx, wy = pixel_to_world(H, px, py)
    print(f"  클릭 픽셀=({px},{py}) -> world=({wx:.1f},{wy:.1f})")
    return Pose(x=wx, y=wy, z=place_z, t=RoArm.GRIPPER_OPEN)


def place_at(arm: RoArm, cap: cv2.VideoCapture, place_pose: Pose, carry_z: float, grip_close_rad: float) -> None:
    high_z = max(place_pose.z, carry_z)
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, grip_close_rad))
    sleep_showing(cap, 1.3)
    arm.move_to(Pose(place_pose.x, place_pose.y, place_pose.z, grip_close_rad))
    sleep_showing(cap, 1.0)
    arm.open_gripper()
    wait_gripper_settled(arm, cap, timeout=2.5)
    sleep_showing(cap, 0.3)
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, RoArm.GRIPPER_OPEN))
    sleep_showing(cap, 0.8)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RoArm-M2 pick-and-place with a fixed overview camera")
    p.add_argument("--target", default=None, help="natural language description of the object (required for --detector vlm)")
    p.add_argument("--detector", choices=("vlm", "darkest", "manual"), default="manual")
    p.add_argument("--vlm-backend", choices=("gemini", "openai"), default=None)
    p.add_argument("--min-area-frac", type=float, default=0.01)
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "/dev/cu.usbserial-120"))
    p.add_argument("--baudrate", type=int, default=115200)
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
    p.add_argument("--click-place-z", type=float, default=None, help="z to set the object down at in --place-mode click (default: calibrated floor z + clearance)")
    p.add_argument("--camera-index", type=int, default=1, help="the fixed overview camera's index (used for detection/homography)")
    p.add_argument("--secondary-camera-index", type=int, default=0, help="extra camera shown live alongside the main one, reference only (no control logic uses it); -1 to disable")
    p.add_argument("--homography", default="ground_homography.json", help="output of calibrate_ground_homography.py")
    p.add_argument("--grid-spacing-mm", type=float, default=50.0, help="AR overlay: grid line spacing in real mm on the calibrated floor area (0 disables the overlay)")
    p.add_argument("--floor-calibration", default="floor_calibration.json")
    p.add_argument("--floor-clearance-mm", type=float, default=5.0, help="gripper stops this far above the calibrated floor")
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

    H, cal_camera_index = load_homography(args.homography)
    if cal_camera_index != args.camera_index:
        print(f"주의: homography는 camera-index {cal_camera_index}로 캘리브레이션됐지만 지금은 {args.camera_index}를 사용합니다.")

    floor_z = load_floor_z(args.floor_calibration)
    grab_z = floor_z + args.floor_clearance_mm
    print(f"floor z={floor_z:.1f}, grab z={grab_z:.1f}")

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
    click_place_z = args.click_place_z if args.click_place_z is not None else grab_z

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {args.camera_index}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    global _secondary_preview
    _secondary_preview = SecondaryPreview(args.secondary_camera_index if args.secondary_camera_index >= 0 else None)

    global _ground_grid
    if args.grid_spacing_mm > 0:
        _ground_grid = GroundGridOverlay(args.homography, grid_spacing_mm=args.grid_spacing_mm)
        if _ground_grid.enabled:
            print(f"AR 그리드 오버레이 켜짐 (간격 {args.grid_spacing_mm:.0f}mm)")

    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"===== cycle {cycle} =====")
            print("moving to start pose")
            arm.move_to(start_pose)
            sleep_showing(cap, 1.0)
            arm.open_gripper()
            wait_gripper_settled(arm, cap, timeout=2.0)

            frame = capture_frame(cap)
            box = detect_target_pixel_box(cap, frame, args.target, args.detector, args.min_area_frac)
            if box is None:
                show_preview(frame, None, None, "no target found")
                if cycle == 1:
                    print("failed to locate target, aborting")
                    return 1
                print("no more objects found; stopping loop")
                break

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            wx, wy = pixel_to_world(H, cx, cy)
            show_preview(frame, box, (wx, wy), f"target pixel=({cx:.0f},{cy:.0f}) -> world=({wx:.1f},{wy:.1f})")
            print(f"target pixel center=({cx:.0f},{cy:.0f}) -> world (x,y)=({wx:.1f},{wy:.1f})")

            approach_pose = Pose(x=wx, y=wy, z=max(grab_z, carry_z), t=RoArm.GRIPPER_OPEN)
            print("moving above target")
            arm.move_to(approach_pose, speed=args.approach_speed)
            sleep_showing(cap, 1.2)

            print("descending to grab height")
            arm.move_to(Pose(wx, wy, grab_z, RoArm.GRIPPER_OPEN), speed=args.approach_speed)
            sleep_showing(cap, 1.0)

            print("picking")
            pick(arm, cap, carry_z, args.grip_settle_sec, args.grip_close_rad)

            if args.place_mode == "click":
                # object is now held, lifted to carry_z -- hover and wait for a click
                print("공중에서 대기 중: 놓을 위치를 화면에서 클릭하세요")
                click_place_pose = pick_place_pose_from_click(cap, H, click_place_z)
                if click_place_pose is None:
                    print("취소됨: 물체를 든 채 시작 자세로 복귀합니다 (그립 유지)")
                    arm.move_to(Pose(start_pose.x, start_pose.y, carry_z, args.grip_close_rad))
                    sleep_showing(cap, 1.5)
                    arm.close_gripper()
                    sleep_showing(cap, 0.3)
                    break
                print(f"placing at clicked point: x={click_place_pose.x:.1f} y={click_place_pose.y:.1f} z={click_place_pose.z:.1f}")
                place_at(arm, cap, click_place_pose, carry_z, args.grip_close_rad)
                print("returning to start pose (gripper open)")
                arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
                sleep_showing(cap, 1.0)
            elif place_pose is None:
                print("returning to start pose, holding the object (no place pose -> single cycle)")
                arm.move_to(start_pose)
                sleep_showing(cap, 1.5)
                arm.close_gripper()
                sleep_showing(cap, 0.3)
                break
            else:
                print("placing at taught pose")
                place_at(arm, cap, place_pose, carry_z, args.grip_close_rad)
                print("returning to start pose (gripper open)")
                arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
                sleep_showing(cap, 1.0)

            if args.max_cycles and cycle >= args.max_cycles:
                print(f"reached max cycles ({args.max_cycles}); stopping")
                break
    except KeyboardInterrupt:
        print("\ninterrupted, stopping")
    finally:
        cap.release()
        cv2.destroyWindow(WINDOW_NAME)
        if _secondary_preview is not None:
            _secondary_preview.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
