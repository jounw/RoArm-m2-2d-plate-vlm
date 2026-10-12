"""Demonstration collector for the FULL end-to-end Diffusion Policy: start
pose all the way to a completed grab, no homography/VLM anywhere. Unlike
dp_collect_data.py (which only records the short descend+grab clip starting
from an already-approached look-down pose), this records the WHOLE
trajectory using BOTH cameras -- the fixed overview camera AND the
eye-in-hand camera -- since the policy now has to do its own rough
approach too, which the fixed camera's wide view is what makes possible.

Each episode:
  1. move the arm to the start pose yourself (or run it there first)
  2. place the object anywhere reachable
  3. press SPACE to start recording
  4. hand-guide (torque OFF) the arm through the ENTIRE trajectory: rough
     approach -> fine positioning -> descent -> grab. Press 1 to close the
     gripper at the moment of grab, 2 to re-open it if you want to retry
     within the same take.
  5. press SPACE again to stop recording (episode saved) once the object
     is gripped and lifted a little

Recorded per timestep, at --freq-hz:
  - fixed camera frame (BGR, resized to --img-size)
  - eye-in-hand camera frame (BGR, resized to --img-size)
  - robot pose feedback (x, y, z, t) via RoArm.feedback() (T:105)

Saved as one HDF5 file per episode in --save-dir (default dp_e2e_dataset),
with datasets `cam_fixed`, `cam_eih`, `pose`, `grip_closed`, `timestamps`
-- this is a different schema from dp_collect_data.py's single-camera
episodes (attrs["stage"] = "e2e_full_trajectory"), read by
dp_dataset_e2e.py, not dp_dataset.py.

Controls: [SPACE] start/stop   [1] close gripper   [2] open gripper
          [D] delete last episode   [Q] quit

Run:
  python3 dp_collect_data_e2e.py --serial-port /dev/cu.usbserial-120 \\
      --fixed-camera-index 1 --camera-index 0 --object red_block
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

from roarm_driver import Pose, RoArm

WINDOW_NAME = "DP-E2E Demo Collection"


def open_camera(index: int, img_w: int, img_h: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {index}")
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
    frames_fixed: list[np.ndarray],
    frames_eih: list[np.ndarray],
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{obj_name}_ep{episode_id:04d}.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["object"] = obj_name
        f.attrs["episode_id"] = episode_id
        f.attrs["collected"] = datetime.now().isoformat()
        f.attrs["freq_hz"] = freq_hz
        f.attrs["stage"] = "e2e_full_trajectory"
        f.create_dataset("timestamps", data=np.array(timestamps, dtype=np.float64))
        f.create_dataset("pose", data=np.array(poses, dtype=np.float32))
        f.create_dataset("grip_closed", data=np.array(grip_closed, dtype=np.bool_))
        f.create_dataset("cam_fixed", data=np.array(frames_fixed, dtype=np.uint8), compression="gzip", compression_opts=4)
        f.create_dataset("cam_eih", data=np.array(frames_eih, dtype=np.uint8), compression="gzip", compression_opts=4)
    return path


def show_preview(frame_fixed: np.ndarray, frame_eih: np.ndarray, status: str, pose: Pose | None) -> None:
    h = 240
    left = cv2.resize(frame_fixed, (320, h))
    right = cv2.resize(frame_eih, (320, h))
    preview = np.hstack([left, right])
    color = (0, 0, 220) if "REC" in status else (180, 180, 180)
    cv2.putText(preview, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(preview, "fixed", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(preview, "eye-in-hand", (330, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if pose is not None:
        txt = f"x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f} t={pose.t:.2f}"
        cv2.putText(preview, txt, (10, h + 20) if preview.shape[0] > h else (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow(WINDOW_NAME, preview)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect FULL start-pose-to-grab demonstrations (both cameras) for end-to-end Diffusion Policy training")
    p.add_argument("--object", required=True)
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "/dev/cu.usbserial-120"))
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--fixed-camera-index", type=int, default=1, help="fixed overview camera index")
    p.add_argument("--camera-index", type=int, default=0, help="eye-in-hand camera index")
    p.add_argument("--img-w", type=int, default=224)
    p.add_argument("--img-h", type=int, default=224)
    p.add_argument("--freq-hz", type=float, default=10.0)
    p.add_argument("--grip-closed-threshold", type=float, default=2.5)
    p.add_argument("--save-dir", default="dp_e2e_dataset")
    return p


def collect_session(args: argparse.Namespace) -> None:
    save_dir = Path(args.save_dir)
    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)
    fixed_cap = open_camera(args.fixed_camera_index, args.img_w, args.img_h)
    eih_cap = open_camera(args.camera_index, args.img_w, args.img_h)

    episode_id = count_episodes(save_dir, args.object)
    print(f"[{args.object}] {episode_id} episodes already saved")
    print("각 에피소드 시작 전: 팔을 START POSE로 옮기고 물체를 작업 범위 아무 데나 놓으세요.")
    print("Controls: [SPACE] 녹화 시작/정지   [1] 그리퍼 닫기   [2] 그리퍼 열기   [D] 마지막 삭제   [Q] 종료\n")

    arm.torque(False)
    print("토크 OFF -- 팔을 손으로 움직여 전체 궤적(대략 접근 -> 미세 접근 -> 하강 -> 그랩)을 시연하세요")

    recording = False
    timestamps: list[float] = []
    poses: list[list[float]] = []
    grip_closed: list[bool] = []
    frames_fixed: list[np.ndarray] = []
    frames_eih: list[np.ndarray] = []
    last_tick = time.time()
    session_count = 0

    try:
        while True:
            now = time.time()
            frame_fixed = read_frame(fixed_cap, args.img_w, args.img_h)
            frame_eih = read_frame(eih_cap, args.img_w, args.img_h)
            fb = arm.feedback()

            status = f"REC ep{episode_id:04d} {len(timestamps)} steps" if recording else f"STANDBY [SPACE] to record  session:{session_count}"
            if frame_fixed is not None and frame_eih is not None:
                show_preview(frame_fixed, frame_eih, status, fb)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not recording:
                    timestamps, poses, grip_closed, frames_fixed, frames_eih = [], [], [], [], []
                    recording = True
                    print(f"  > ep{episode_id:04d} 시작")
                else:
                    recording = False
                    if len(timestamps) < max(3, int(args.freq_hz)):
                        print(f"  x 너무 짧음 ({len(timestamps)} steps), 버림")
                    else:
                        path = save_episode(save_dir, args.object, episode_id, args.freq_hz, timestamps, poses, grip_closed, frames_fixed, frames_eih)
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

            if recording and frame_fixed is not None and frame_eih is not None and fb is not None and (now - last_tick) >= 1.0 / args.freq_hz:
                timestamps.append(now)
                poses.append([fb.x, fb.y, fb.z, fb.t])
                grip_closed.append(bool(fb.t >= args.grip_closed_threshold))
                frames_fixed.append(frame_fixed)
                frames_eih.append(frame_eih)
                last_tick = now
    finally:
        arm.torque(True)
        fixed_cap.release()
        eih_cap.release()
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
