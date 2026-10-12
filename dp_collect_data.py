"""Demonstration collector for the Diffusion Policy stage ONLY: eye-in-hand
precise descent + grab. The classical fixed-camera rough approach stays
exactly as-is (calibrate_ground_homography.py / pick_place_fixed_cam.py) --
this script is not meant to teach that part, only the part that
track_and_descend currently does by VLM+tracker+geometry.

Each episode should start from roughly the "look-down" pose that
pick_place_fixed_to_eih.py already puts the arm at right before calling
track_and_descend (a fixed x,y roughly above an object, some clearance
above the floor -- see --lookdown-clearance-mm there), with an object placed
somewhere under/near the gripper. From there you hand-guide (torque OFF)
the arm through fine positioning and the grab itself, ending with the
gripper closed on the object.

Recorded per timestep, at --freq-hz:
  - eye-in-hand camera frame (BGR, resized to --img-size)
  - robot pose feedback (x, y, z, t) via RoArm.feedback() (T:105)
Saved as one HDF5 file per episode in --save-dir, matching the general
shape/convention of data/collect_data.py (timestamps + per-step arrays,
gzip-compressed image dataset) but for a single eye-in-hand camera and
continuous (x,y,z,t) pose instead of 4 joint angles + dual camera, since
that is what pick_place_fixed_to_eih.py's action space actually is
(move_to(Pose(x,y,z,t))).

Controls: [SPACE] start/stop recording   [1] close gripper   [2] open gripper
          [D] delete last episode   [Q] quit

The gripper (EOAT, T:106) is driven by keyboard, not by hand, even while
--torque is OFF for the arm joints: RoArm-M2's torque toggle (T:210) only
frees the main arm servos so you can pose it by hand, the gripper servo
stays independently commandable. So during a demo you hand-guide position
with your other hand and press 1 to close the gripper at the moment of
grab -- that's what actually produces a realistic, sharp grip_closed
transition in the recorded pose.t stream (dragging the gripper open/closed
by hand against a live servo is unreliable/awkward on this hardware).

Run:
  python3 dp_collect_data.py --serial-port /dev/cu.usbserial-120 --camera-index 0 --object "red_block"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

from roarm_driver import Pose, RoArm


def open_camera(index: int, img_w: int, img_h: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open eye-in-hand camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)
    return cap


def read_frame(cap: cv2.VideoCapture, img_w: int, img_h: int) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok:
        return None
    if frame.shape[1] != img_w or frame.shape[0] != img_h:
        frame = cv2.resize(frame, (img_w, img_h))
    return frame


def count_episodes(save_dir: Path, obj_name: str) -> int:
    return len(list(save_dir.glob(f"{obj_name}_ep*.hdf5")))


def save_episode(
    save_dir: Path,
    obj_name: str,
    episode_id: int,
    freq_hz: float,
    timestamps: list[float],
    poses: list[list[float]],
    grip_closed: list[bool],
    frames: list[np.ndarray],
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{obj_name}_ep{episode_id:04d}.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["object"] = obj_name
        f.attrs["episode_id"] = episode_id
        f.attrs["collected"] = datetime.now().isoformat()
        f.attrs["freq_hz"] = freq_hz
        f.attrs["stage"] = "eih_descend_and_grab"
        f.create_dataset("timestamps", data=np.array(timestamps, dtype=np.float64))
        # pose = (x, y, z, t) in the same units/convention as roarm_driver.Pose /
        # RoArm.move_to -- this IS the action space the policy will output.
        f.create_dataset("pose", data=np.array(poses, dtype=np.float32))
        f.create_dataset("grip_closed", data=np.array(grip_closed, dtype=np.bool_))
        f.create_dataset(
            "cam_eih", data=np.array(frames, dtype=np.uint8), compression="gzip", compression_opts=4
        )
    return path


def show_preview(frame: np.ndarray, status: str, pose: Pose | None) -> None:
    disp = frame.copy()
    color = (0, 0, 220) if "REC" in status else (180, 180, 180)
    cv2.putText(disp, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if pose is not None:
        txt = f"x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f} t={pose.t:.2f}"
        cv2.putText(disp, txt, (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("DP Demo Collection (eye-in-hand)", disp)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect eye-in-hand descend+grab demonstrations for Diffusion Policy training")
    p.add_argument("--object", required=True, help="object/task name; episodes are grouped and named by this")
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "/dev/cu.usbserial-120"))
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--camera-index", type=int, default=0, help="eye-in-hand camera index")
    p.add_argument("--img-w", type=int, default=224)
    p.add_argument("--img-h", type=int, default=224)
    p.add_argument("--freq-hz", type=float, default=10.0)
    p.add_argument("--grip-closed-threshold", type=float, default=2.5, help="pose.t above this is considered 'gripper closed' for the recorded grip_closed flag (GRIPPER_OPEN=1.57, GRIPPER_CLOSED=3.14)")
    p.add_argument("--save-dir", default="dp_dataset")
    return p


def collect_session(args: argparse.Namespace) -> None:
    save_dir = Path(args.save_dir)
    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)
    cap = open_camera(args.camera_index, args.img_w, args.img_h)

    episode_id = count_episodes(save_dir, args.object)
    print(f"[{args.object}] {episode_id} episodes already saved")
    print("각 에피소드 시작 전: 팔을 look-down pose 근처(고정캠 대략 접근이 끝난 위치)에 두고 물체를 그 아래 놓으세요.")
    print("Controls: [SPACE] 녹화 시작/정지   [1] 그리퍼 닫기   [2] 그리퍼 열기   [D] 마지막 삭제   [Q] 종료\n")

    arm.torque(False)
    print("토크 OFF -- 팔을 손으로 움직여 시연하세요 (녹화 시작 후 미세 접근 -> 그랩까지)")

    recording = False
    timestamps: list[float] = []
    poses: list[list[float]] = []
    grip_closed: list[bool] = []
    frames: list[np.ndarray] = []
    last_tick = time.time()
    session_count = 0

    try:
        while True:
            now = time.time()
            frame = read_frame(cap, args.img_w, args.img_h)
            fb = arm.feedback()

            status = f"REC ep{episode_id:04d} {len(timestamps)} steps" if recording else f"STANDBY [SPACE] to record  session:{session_count}"
            if frame is not None:
                show_preview(frame, status, fb)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not recording:
                    timestamps, poses, grip_closed, frames = [], [], [], []
                    recording = True
                    print(f"  > ep{episode_id:04d} 시작")
                else:
                    recording = False
                    if len(timestamps) < max(3, int(args.freq_hz)):
                        print(f"  x 너무 짧음 ({len(timestamps)} steps), 버림")
                    else:
                        path = save_episode(save_dir, args.object, episode_id, args.freq_hz, timestamps, poses, grip_closed, frames)
                        print(f"  + 저장됨: {path.name} ({len(timestamps)} steps)")
                        episode_id += 1
                        session_count += 1
            elif key == ord("1"):
                arm.close_gripper()
                print("  그리퍼 닫기")
            elif key == ord("2"):
                arm.open_gripper()
                print("  그리퍼 열기")
            elif key == ord("d"):
                p = save_dir / f"{args.object}_ep{episode_id - 1:04d}.hdf5"
                if p.exists():
                    p.unlink()
                    episode_id -= 1
                    session_count = max(0, session_count - 1)
                    print(f"  삭제됨: {p.name}")
                else:
                    print("  삭제할 게 없습니다")
            elif key == ord("q"):
                break

            if recording and frame is not None and fb is not None and (now - last_tick) >= 1.0 / args.freq_hz:
                timestamps.append(now)
                poses.append([fb.x, fb.y, fb.z, fb.t])
                grip_closed.append(bool(fb.t >= args.grip_closed_threshold))
                frames.append(frame)
                last_tick = now
    finally:
        arm.torque(True)
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n세션 종료 -- 이번 세션 {session_count}개, 총 {episode_id}개")


def print_stats(save_dir: Path, obj_name: str) -> None:
    n = count_episodes(save_dir, obj_name)
    print(f"[{obj_name}] {n} episodes in {save_dir}/")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print_stats(Path(args.save_dir), args.object)
    collect_session(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
