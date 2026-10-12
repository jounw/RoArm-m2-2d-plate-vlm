"""RoArm-M2 language-commanded pick-and-place from a single top-down camera.

A fixed overhead camera (--topdown: an Iriun-connected phone webcam by
default, or an iPad via Record3D -- RGB only, depth never read) sees the
whole workspace. Each cycle:
  1. get a command (typed / spoken via --voice / fixed --target / manual box)
  2. VLM parses the utterance against the current frame into a pick/place
     intent (scene questions are answered directly, no motion)
  3. ONE VLM call grounds both the object and the destination as boxes
  4. box centers -> robot (x, y) via the ground-plane homography;
     z comes from the floor calibration
  5. hover approach, diagonal descent to grab height, grasp, place, home
No eye-in-hand camera and no depth sensing anywhere.

Rotation: the preview may be rotated for display (--rotate) but detection
boxes / homography pixels always live in the ORIGINAL unrotated frame. Use
the same rotation the homography was calibrated with.

Run:
  # 1) one-time calibration (if not done yet, in this order):
  python3 teach_start_pose.py --serial-port /dev/cu.usbserial-120
  python3 calibrate_workspace.py --serial-port /dev/cu.usbserial-120
  python3 calibrate_floor.py --serial-port /dev/cu.usbserial-120
  python3 calibrate_ground_homography.py --camera-index 0 --serial-port /dev/cu.usbserial-120 --num-points 9

  # 2) natural-language commands (text; add --voice for mic + TTS):
  python3 pick_place_record3d_to_eih.py --place-mode command --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import pick_place as _pp
import pick_place_record3d_rgb as _r3
from pick_place import (
    WORKSPACE,
    clamp,
    expand_workspace,
    load_place_pose,
    load_start_pose,
    load_workspace_calibration,
)
from pick_place_record3d_rgb import (
    GroundGridOverlay,
    detect_target_pixel_box,
    load_homography,
    normalized_box_to_pixel,
    pick_place_pose_from_click,
    pixel_to_world,
    show_preview,
    world_to_pixel,
)
from record3d_camera import Record3DCamera, rotate_for_display
from roarm_driver import Pose, RoArm
from vision import locate, locate_many, parse_pick_place

R3_WINDOW_NAME = _r3.WINDOW_NAME  # top-down window (shared with pick_place_record3d_rgb helpers)


class _WebcamFrame:
    """Duck-typed stand-in for record3d_camera.Frame (rgb only)."""

    def __init__(self, rgb: np.ndarray):
        self.rgb = rgb
        self.depth_m = None


class WebcamTopdown:
    """Plain cv2.VideoCapture (e.g. an Iriun-connected phone) exposed through
    the same get_frame()/close() interface as Record3DCamera, so the rest of
    the pipeline doesn't care which top-down camera is plugged in."""

    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open top-down webcam index {index}")

    def get_frame(self, timeout: float = 5.0) -> _WebcamFrame:
        deadline = time.time() + timeout
        while time.time() < deadline:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return _WebcamFrame(frame)
        raise RuntimeError("failed to read a top-down webcam frame")

    def close(self) -> None:
        self.cap.release()


class RecordingRecord3D:
    """Top-down camera wrapper whose recording captures the DISPLAYED view --
    HUD, grid, markers, labels and all -- not the raw camera frames:
    cv2.imshow is hooked (see main()) so every frame shown in the top-down
    window is also written to the video via write_display()."""

    def __init__(self, cam, record: bool = True, out_dir: str = "recordings", fps: float = 20.0):
        self.cam = cam
        self.fps = fps
        self.writer = None
        self.path = None
        self.video_t0: float | None = None
        self._latest_disp = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._writer_thread = None
        if record:
            Path(out_dir).mkdir(exist_ok=True)
            self.path = str(Path(out_dir) / time.strftime("record3d_%Y%m%d_%H%M%S.mp4"))
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()

    def get_frame(self, timeout: float = 5.0):
        return self.cam.get_frame(timeout)

    def write_display(self, disp: np.ndarray) -> None:
        """Called (via the cv2.imshow hook) with every displayed frame; the
        writer thread paces the actual writes by wall-clock time."""
        with self._lock:
            self._latest_disp = disp.copy()

    def _writer_loop(self) -> None:
        # Wall-clock paced: exactly `fps` frames per real
        # second (repeating the latest displayed frame while the main thread
        # is busy), so the muxed-in audio stays in sync.
        written = 0
        while not self._stop.is_set():
            time.sleep(1.0 / self.fps / 2)
            with self._lock:
                disp = self._latest_disp
            if disp is None:
                continue
            now = time.time()
            if self.writer is None:
                h, w = disp.shape[:2]
                self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                self.video_t0 = now
            target = int((now - self.video_t0) * self.fps) + 1
            while written < target:
                self.writer.write(disp)
                written += 1

    def close(self) -> None:
        self.cam.close()
        self._stop.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2.0)
        if self.writer is not None:
            self.writer.release()
            print(f"Record3D 영상 저장됨 (오버레이 포함): {self.path}")


class AudioRecorder:
    """Records the default mic to a wav for the whole run, so it can be muxed
    into the videos afterwards (the mic hears both the user's spoken commands
    and the robot's TTS replies from the speakers). Failure to open the mic
    never kills the run -- recording is just skipped."""

    def __init__(self, path: str, samplerate: int | None = None, device: str | None = None):
        self.path = path
        self.t0: float | None = None
        self._chunks: list[np.ndarray] = []
        self._stream = None
        try:
            import sounddevice as sd

            # resolve the mic: an index, a name substring, or the system default.
            # The system default silently changes (e.g. AirPods connecting) and a
            # pocketed AirPods mic hears nothing -- so always SAY which mic we use.
            dev = None
            if device is not None:
                if device.isdigit():
                    dev = int(device)
                else:
                    for i, d in enumerate(sd.query_devices()):
                        if d["max_input_channels"] > 0 and device.lower() in d["name"].lower():
                            dev = i
                            break
                    if dev is None:
                        raise RuntimeError(f"no input device matching {device!r}")
            info = sd.query_devices(dev, "input")
            if samplerate is None:
                # use the device's native rate -- forcing 16k on a 48k device
                # is what triggers PaMacCore/AUHAL err -50 on macOS
                samplerate = int(info["default_samplerate"])
            self.samplerate = samplerate
            print(f"(마이크: {info['name']} @ {samplerate}Hz)")
            self._stream = sd.InputStream(
                device=dev, samplerate=samplerate, channels=1, dtype="int16",
                callback=lambda indata, frames, t, status: self._chunks.append(indata.copy()),
            )
            self._stream.start()
            self.t0 = time.time()
        except Exception as e:
            print(f"(마이크 녹음 실패, 영상은 무음으로 저장됩니다: {e})")
            self._stream = None
            self.samplerate = samplerate or 16000

    @property
    def active(self) -> bool:
        return self._stream is not None

    def _drain(self, consumed: int) -> tuple[int, list[np.ndarray]]:
        new = []
        while consumed < len(self._chunks):
            new.append(self._chunks[consumed].flatten())
            consumed += 1
        return consumed, new

    def listen(self, max_seconds: float = 15.0, silence_duration: float = 1.2,
               min_speech_duration: float = 0.4, pre_roll_sec: float = 0.3) -> np.ndarray | None:
        """Voice-command capture THROUGH the already-open recording stream --
        opening a second InputStream on the same mic (what speech.listen()
        does) makes macOS AUHAL fail with err -50.

        The speech/silence threshold is ADAPTIVE: ambient noise is sampled for
        ~0.5 s at the start of every listen and the threshold is set well
        above it, so a fan or hum never counts as speech and normal speech
        always does. A short pre-roll is kept so the first syllable isn't
        clipped. Returns int16 mono audio at self.samplerate, or None."""
        if not self.active:
            return None

        # 1) sample the ambient noise floor
        consumed = len(self._chunks)
        levels: list[float] = []
        t_end = time.time() + 0.5
        while time.time() < t_end:
            consumed, new = self._drain(consumed)
            levels += [float(np.abs(c).mean()) for c in new]
            time.sleep(0.02)
        noise = float(np.median(levels)) if levels else 100.0
        threshold = max(350.0, noise * 3.0)
        print(f"(듣는 중... 말씀하세요 — 주변소음 {noise:.0f}, 음성 임계값 {threshold:.0f})")

        # 2) wait for speech, record until silence
        frames: list[np.ndarray] = []
        pre_roll: deque[np.ndarray] = deque(maxlen=max(1, int(pre_roll_sec * self.samplerate / 512)))
        speech_started = False
        silent_samples = 0
        silence_limit = int(silence_duration * self.samplerate)
        deadline = time.time() + max_seconds
        while time.time() < deadline:
            consumed, new = self._drain(consumed)
            for chunk in new:
                loud = float(np.abs(chunk).mean()) > threshold
                if not speech_started:
                    pre_roll.append(chunk)
                    if loud:
                        speech_started = True
                        frames.extend(pre_roll)  # keep the syllable onset
                        print("(녹음 중...)")
                        deadline = time.time() + max_seconds  # full budget for the utterance itself
                else:
                    frames.append(chunk)
                    if loud:
                        silent_samples = 0
                    else:
                        silent_samples += len(chunk)
                        if silent_samples >= silence_limit:
                            audio = np.concatenate(frames)
                            dur = len(audio) / self.samplerate
                            if dur < min_speech_duration:
                                print(f"(너무 짧음: {dur:.2f}s, 무시)")
                                return None
                            print(f"(녹음 완료: {dur:.1f}s)")
                            return audio
            time.sleep(0.02)
        if not frames:
            return None
        audio = np.concatenate(frames)
        if len(audio) / self.samplerate < min_speech_duration:
            return None
        print(f"(녹음 완료: {len(audio) / self.samplerate:.1f}s, 최대 시간 도달)")
        return audio

    def stop(self) -> str | None:
        """Stop and write the wav; returns its path (None if nothing recorded)."""
        if self._stream is None:
            return None
        import wave

        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks)
        with wave.open(self.path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(audio.tobytes())
        return self.path


def mux_audio(video_path: str | None, video_t0: float | None, audio_path: str, audio_t0: float) -> None:
    """Merge the run's mic recording into a finished video with ffmpeg,
    trimming the audio's start so it lines up with when the video began."""
    import shutil
    import subprocess

    if video_path is None or video_t0 is None or not Path(video_path).exists():
        return
    if shutil.which("ffmpeg") is None:
        print(f"(ffmpeg가 없어 오디오를 합치지 못했습니다; 오디오는 {audio_path}에 따로 있습니다)")
        return
    offset = max(0.0, video_t0 - audio_t0)
    tmp_out = video_path + ".mux.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-ss", f"{offset:.3f}", "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        tmp_out,
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp_out, video_path)
        print(f"오디오 합쳐짐: {video_path}")
    except subprocess.CalledProcessError as e:
        print(f"(오디오 합치기 실패: {e}; 오디오는 {audio_path}에 따로 있습니다)")
        Path(tmp_out).unlink(missing_ok=True)


class TopdownHUD:
    """Live telemetry overlay for the Record3D top-down window: the arm's
    real-time pose (projected onto the floor via the inverse homography),
    its recent trajectory, the current target, gripper state, phase, cycle,
    elapsed time and FPS. Pose comes from arm.feedback(), polled (throttled)
    while the overlay runs -- so the marker moves with the real arm."""

    FEEDBACK_INTERVAL = 0.12  # s between serial feedback polls

    def __init__(self, H: np.ndarray, rotate: str):
        self.H_inv = np.linalg.inv(H)
        self.rotate = rotate
        self.phase = "init"
        self.cycle = 0
        self.target_box: tuple[int, int, int, int] | None = None
        self.target_world: tuple[float, float] | None = None
        self.arm_pose: Pose | None = None
        self.trail: deque[tuple[float, float]] = deque(maxlen=200)
        self.t0 = time.time()
        self._last_fb = 0.0
        self._last_frame = None
        self.fps = 0.0

    def poll_arm(self, arm: RoArm) -> None:
        now = time.time()
        if now - self._last_fb < self.FEEDBACK_INTERVAL:
            return
        self._last_fb = now
        fb = arm.feedback()
        if fb is not None:
            self.arm_pose = fb
            self.trail.append((fb.x, fb.y))

    def _tick_fps(self) -> None:
        now = time.time()
        if self._last_frame is not None:
            dt = now - self._last_frame
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        self._last_frame = now

    def _to_display_xy(self, px: int, py: int, orig_w: int, orig_h: int) -> tuple[int, int]:
        """original-frame pixel -> pixel on the rotated display (inverse of
        record3d_camera.displayed_to_original_xy)."""
        if self.rotate == "left":
            return py, orig_w - 1 - px
        if self.rotate == "right":
            return orig_h - 1 - py, px
        if self.rotate == "180":
            return orig_w - 1 - px, orig_h - 1 - py
        return px, py

    def render(self, rgb: np.ndarray) -> np.ndarray:
        self._tick_fps()
        annotated = rgb.copy()
        orig_h, orig_w = rgb.shape[:2]
        _r3._draw_grid(annotated)

        # arm trajectory ribbon (floor projection), fading with age
        pts = [world_to_pixel(self.H_inv, x, y) for x, y in self.trail]
        n = len(pts)
        for i in range(1, n):
            fade = 0.3 + 0.7 * i / n
            cv2.line(annotated, pts[i - 1], pts[i], (0, int(200 * fade), int(255 * fade)), 2, cv2.LINE_AA)

        # target (shapes only -- text labels go on the rotated view below so
        # they read upright)
        tpx = None
        if self.target_box is not None:
            x1, y1, x2, y2 = self.target_box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 80, 80), 2)
            tpx = ((x1 + x2) // 2, (y1 + y2) // 2)
        elif self.target_world is not None:
            tpx = world_to_pixel(self.H_inv, *self.target_world)
        if tpx is not None:
            cv2.drawMarker(annotated, tpx, (255, 80, 80), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)

        # gripper marker (arm x/y projected to the floor) + line to target
        gpx = None
        dist_mm = None
        if self.arm_pose is not None:
            gpx = world_to_pixel(self.H_inv, self.arm_pose.x, self.arm_pose.y)
            cv2.circle(annotated, gpx, 10, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(annotated, gpx, 2, (0, 255, 0), -1, cv2.LINE_AA)
            if tpx is not None and self.target_world is not None:
                cv2.line(annotated, gpx, tpx, (0, 255, 255), 1, cv2.LINE_AA)
                dist_mm = ((self.arm_pose.x - self.target_world[0]) ** 2 + (self.arm_pose.y - self.target_world[1]) ** 2) ** 0.5

        disp = rotate_for_display(annotated, self.rotate).copy()

        # point labels, upright on the rotated view
        if tpx is not None and self.target_world is not None:
            dx, dy = self._to_display_xy(*tpx, orig_w, orig_h)
            _r3.draw_label(disp, (dx + 14, dy + 24), f"TARGET ({self.target_world[0]:.0f},{self.target_world[1]:.0f})mm")
        if gpx is not None:
            dx, dy = self._to_display_xy(*gpx, orig_w, orig_h)
            _r3.draw_label(disp, (dx + 14, dy - 16), f"ARM z={self.arm_pose.z:.0f}", color=(0, 255, 0))

        # telemetry panel (drawn on the rotated view so the text reads upright)
        p = self.arm_pose
        grip = "-" if p is None else ("CLOSED" if p.t > 2.5 else "OPEN")
        elapsed = time.time() - self.t0
        lines = [
            f"RoArm-M2 PICK & PLACE   cycle {self.cycle}   {int(elapsed // 60):02d}:{int(elapsed % 60):02d}   {self.fps:4.1f} fps",
            f"PHASE  {self.phase}",
            "ARM    (waiting for feedback)" if p is None else f"ARM    x{p.x:+8.1f}  y{p.y:+8.1f}  z{p.z:+8.1f} mm",
            f"GRIP   {grip}" + ("" if p is None else f"  ({p.t:.2f} rad)"),
        ]
        if self.target_world is not None:
            t = f"TARGET x{self.target_world[0]:+8.1f}  y{self.target_world[1]:+8.1f} mm"
            if dist_mm is not None:
                t += f"   dist {dist_mm:6.1f} mm"
            lines.append(t)

        pad, lh = 10, 24
        panel_w = 8 + max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for l in lines) + 2 * pad
        panel_h = lh * len(lines) + 2 * pad
        overlay = disp.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.65, disp, 0.35, 0, dst=disp)
        for i, line in enumerate(lines):
            color = (0, 255, 0) if line.startswith("ARM") else (240, 240, 240)
            cv2.putText(disp, line, (8 + pad, 8 + pad + lh * (i + 1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return disp


def show_status(cam, arm: RoArm, hud: TopdownHUD, seconds: float) -> None:
    """Live top-down view with the HUD for `seconds` -- used during every arm
    motion/wait so the overhead window never freezes. cam.get_frame() also
    refreshes the eye-in-hand window (see RecordingRecord3D)."""
    end = time.time() + seconds
    while time.time() < end:
        frame = cam.get_frame()
        hud.poll_arm(arm)
        cv2.imshow(R3_WINDOW_NAME, hud.render(frame.rgb))
        cv2.waitKey(15)


def wait_gripper_settled_hud(arm: RoArm, cam, hud: TopdownHUD, timeout: float = 2.5) -> None:
    deadline = time.time() + timeout
    last_t = None
    while time.time() < deadline:
        show_status(cam, arm, hud, 0.15)
        fb = arm.feedback()
        if fb is None:
            continue
        if last_t is not None and abs(fb.t - last_t) < 0.01:
            return
        last_t = fb.t


def pick_hud(arm: RoArm, cam, hud: TopdownHUD, carry_z: float, grip_settle_sec: float, grip_close_rad: float) -> None:
    hud.phase = "grab (closing gripper)"
    arm.set_gripper(grip_close_rad)
    wait_gripper_settled_hud(arm, cam, hud, timeout=3.0)
    show_status(cam, arm, hud, grip_settle_sec)
    hud.phase = "lift"
    pose = arm.feedback()
    if pose is not None:
        pose.z = max(pose.z, carry_z)
        arm.move_to(pose)
    show_status(cam, arm, hud, 1.0)


def place_at_hud(arm: RoArm, cam, hud: TopdownHUD, place_pose: Pose, carry_z: float, grip_close_rad: float) -> None:
    high_z = max(place_pose.z, carry_z)
    hud.phase = "carry to place"
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, grip_close_rad))
    show_status(cam, arm, hud, 1.3)
    hud.phase = "lower to place"
    arm.move_to(Pose(place_pose.x, place_pose.y, place_pose.z, grip_close_rad))
    show_status(cam, arm, hud, 1.0)
    hud.phase = "release"
    arm.open_gripper()
    wait_gripper_settled_hud(arm, cam, hud, timeout=2.5)
    show_status(cam, arm, hud, 0.3)
    hud.phase = "retract"
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, RoArm.GRIPPER_OPEN))
    show_status(cam, arm, hud, 0.8)


def build_parser():
    p = _pp.build_parser()
    p.description = "RoArm-M2 pick-and-place: top-down camera + ground-plane homography (no eye-in-hand camera)"
    # --floor-clearance-mm default is raised (5 -> 15): stop the descent a bit
    # higher above the calibrated floor before grabbing.
    for action in p._actions:
        if action.dest == "floor_clearance_mm":
            action.default = 15.0
    p.add_argument(
        "--topdown",
        choices=("webcam", "record3d"),
        default="webcam",
        help="top-down overview camera: 'webcam' = plain cv2 camera such as an Iriun-connected phone (default), 'record3d' = iPad via the Record3D app",
    )
    p.add_argument("--topdown-camera-index", type=int, default=0, help="--topdown webcam: camera index of the top-down webcam (e.g. Iriun)")
    p.add_argument("--device-index", type=int, default=0, help="--topdown record3d: Record3D device index, if multiple iOS devices are connected")
    p.add_argument(
        "--rotate",
        choices=("none", "left", "right", "180"),
        default=None,
        help="rotate the top-down preview (display only; must match how the homography was calibrated). Default: 'none' for --topdown webcam, 'left' for record3d",
    )
    p.add_argument("--homography", default=None, help="ground-plane homography JSON. Default: ground_homography.json for --topdown webcam (calibrate_ground_homography.py), ground_homography_record3d.json for record3d")
    p.add_argument("--grid-spacing-mm", type=float, default=50.0, help="AR overlay on the Record3D window: grid spacing in real mm (0 disables)")
    p.add_argument("--hover-clearance-mm", type=float, default=120.0, help="height above the calibrated floor to stop the rough approach at")
    p.add_argument("--descend-speed", type=float, default=0.4, help="arm speed (0-1) for the hover -> grab-height descent")
    p.add_argument("--mic", default="MacBook", help="voice mode input device: index or name substring (default 'MacBook' = built-in mic; use '' for the system default, which may silently be AirPods/Iriun)")
    p.add_argument("--grab-offset-x-mm", type=float, default=0.0, help="fixed x (forward/back) correction folded into the descent")
    p.add_argument("--grab-offset-y-mm", type=float, default=0.0, help="fixed y (lateral) correction folded into the descent; e.g. +50 shifts ~5cm toward +y ('left')")
    p.add_argument(
        "--place-mode",
        choices=("taught", "click", "command"),
        default="taught",
        help="'taught' uses --place-pose/place_pose.json; 'click' waits for you to click the drop point on the Record3D window each cycle; "
             "'command' asks for a natural-language command each cycle ('X 잡아서 3에 놔' / 'X 잡아' / '3에 놔') and uses the VLM to find both "
             "the object and the numbered place spot on the top-down view -- pick-only commands grab and hold until the next command",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.vlm_backend:
        os.environ["VLM_BACKEND"] = args.vlm_backend

    # source-dependent defaults: a plain webcam is upright and calibrated with
    # calibrate_ground_homography.py; Record3D comes out sideways
    if args.rotate is None:
        args.rotate = "none" if args.topdown == "webcam" else "left"
    if args.homography is None:
        args.homography = "ground_homography.json" if args.topdown == "webcam" else "ground_homography_record3d.json"

    H = load_homography(args.homography)

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
            print("no taught place pose found; will hold the object at the start pose (run teach_place.py to set one)")
    elif args.place_mode == "click":
        print("place-mode=click: 매 사이클 물체를 든 채 대기하다가, Record3D 화면 클릭으로 놓을 위치를 지정합니다.")
    else:
        print("place-mode=command: 매 사이클 자연어 명령('X 잡아서 3에 놔' / 'X 잡아' / '2에 놔')을 입력받아 VLM으로 물체와 숫자 위치를 찾습니다.")

    floor_z = _pp.load_floor_z(args.floor_calibration)
    if floor_z is not None:
        print(f"floor calibration loaded: z={floor_z:.1f} (from {args.floor_calibration})")
    else:
        print(f"no floor calibration found at {args.floor_calibration}; using clearances relative to the start pose instead")

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

    hover_z = clamp(floor_z + args.hover_clearance_mm, "z") if floor_z is not None else clamp(carry_z, "z")
    print(f"approach hover height: {hover_z:.1f}")

    click_place_z = clamp(floor_z + args.floor_clearance_mm, "z") if floor_z is not None else clamp(start_pose.z - args.place_down_mm, "z")

    grab_z = clamp(floor_z + args.floor_clearance_mm, "z") if floor_z is not None else clamp(start_pose.z - args.pick_down_mm, "z")
    print(f"grab height: {grab_z:.1f}")

    # top-down overview camera (Iriun webcam or Record3D)
    if args.topdown == "webcam":
        print(f"탑다운 웹캠(인덱스 {args.topdown_camera_index}) 연결 중... (Iriun 사용 시 폰에서 Iriun Webcam 앱이 켜져 있어야 함)")
        base_cam = WebcamTopdown(args.topdown_camera_index)
    else:
        print("Record3D 카메라에 연결 중... (iPad에서 Record3D 앱 -> USB Streaming mode 켜져 있어야 함, RGB만 사용합니다)")
        base_cam = Record3DCamera(device_index=args.device_index)
    cam = RecordingRecord3D(base_cam, record=args.record)
    if cam.path is not None:
        print(f"Record3D 영상 녹화 중 (화면 표시 그대로, 오버레이 포함): {cam.path}")

        # hook cv2.imshow so EVERYTHING shown in the Record3D window -- HUD,
        # grid, boxes, labels, prompts (including frames drawn by the reused
        # pick_place_record3d_rgb interaction loops) -- lands in the recording
        _orig_imshow = cv2.imshow

        def _imshow_recording(winname, mat):
            if winname == R3_WINDOW_NAME:
                cam.write_display(mat)
            _orig_imshow(winname, mat)

        cv2.imshow = _imshow_recording
    cv2.namedWindow(R3_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.grid_spacing_mm > 0:
        _r3._ground_grid = GroundGridOverlay(args.homography, grid_spacing_mm=args.grid_spacing_mm)
        if _r3._ground_grid.enabled:
            print(f"AR 그리드 오버레이 켜짐 (간격 {args.grid_spacing_mm:.0f}mm)")

    # run-long mic recording, muxed into the video on exit
    audio = None
    if args.record:
        Path("recordings").mkdir(exist_ok=True)
        audio = AudioRecorder(str(Path("recordings") / time.strftime("audio_%Y%m%d_%H%M%S.wav")), device=args.mic or None)
        if audio.active:
            print("마이크 오디오 녹음 중 (종료 시 영상에 합쳐집니다)")

    executor = ThreadPoolExecutor(max_workers=2)  # background VLM grounding + misc

    speech_mod = None
    say_queue: queue.Queue[str] = queue.Queue()
    if args.voice:
        # imported lazily: sounddevice/OpenAI-key only needed when voice is on
        import speech as speech_mod

        print("voice mode: 마이크로 명령을 말하면 whisper로 받아적고, 응답은 TTS로 말합니다")

        def _speech_worker() -> None:
            # TTS runs in the background so speaking never blocks arm motion
            # or the VLM calls; a single worker keeps utterances in order
            while True:
                text = say_queue.get()
                try:
                    speech_mod.speak(text)
                except Exception as e:
                    print(f"(TTS 실패: {e})")
                finally:
                    say_queue.task_done()

        threading.Thread(target=_speech_worker, daemon=True).start()

    def say(text: str) -> None:
        """Spoken feedback when --voice is on (asynchronous; speak() prints the
        text when it actually plays); no-op otherwise."""
        if speech_mod is not None:
            say_queue.put(text)

    hud = TopdownHUD(H, args.rotate)
    holding = False  # command mode: a pick-only command grabs and holds until a place command
    holding_desc = None  # what the gripper is holding, for context-aware parsing
    dialog: list[str] = []  # rolling dialogue/action transcript fed back to the parser

    def log_dialog(entry: str) -> None:
        dialog.append(entry)
        del dialog[:-16]

    try:
        cycle = 0
        while True:
            cycle += 1
            hud.cycle = cycle
            hud.target_box = None
            hud.target_world = None
            hud.trail.clear()
            print(f"===== cycle {cycle} =====")

            # ---- command mode: get and parse a natural-language command ----
            pick_desc = args.target
            place_desc = None
            cmd_place_pose = None  # pre-located place spot ("X 잡아서 N에 놔": both boxes come from ONE VLM call)
            cmd_frame = None  # frame the command was parsed (and grounded) against
            ground_future = None  # VLM grounding running in the background while the arm moves
            if args.place_mode == "command":
                hud.phase = "waiting for command"
                show_status(cam, arm, hud, 0.2)
                held = " [물체를 잡고 있음]" if holding else ""
                if args.voice:
                    print(f"\n음성 명령{held} (예: '검은 물체 잡아서 3에 놔' / '펜 잡아' / '2에 놔', '종료'라고 말하면 끝)")
                    say_queue.join()  # 스피커 TTS가 끝난 뒤에 들어야 로봇이 자기 목소리를 받아적지 않음
                    utterance = None
                    while not utterance:
                        if audio is not None and audio.active:
                            # tap the run-long recording stream -- a second
                            # InputStream on the same mic breaks macOS AUHAL
                            seg = audio.listen()
                            utterance = (
                                speech_mod.transcribe(
                                    seg, samplerate=audio.samplerate,
                                    prompt="로봇 팔 명령. 예: 검은 물체 잡아서 3에 놔. 테이프 잡아. 2에 놔. 테이블에 뭐 보여? 종료.",
                                )
                                if seg is not None else None
                            )
                        else:
                            utterance = speech_mod.listen()
                        if utterance is None:
                            print("(아무 소리도 못 들었습니다. 다시 말씀해주세요)")
                    utterance = utterance.strip()
                    print(f"[나] {utterance}")
                    if any(w in utterance for w in ("종료", "그만", "끝내")):
                        say("종료합니다")
                        break
                else:
                    utterance = input(f"\n명령 입력{held} (예: '검은 물체 잡아서 3에 놔' / '펜 잡아' / '2에 놔', 빈 입력=종료): ").strip()
                    if not utterance:
                        print("종료합니다")
                        break
                cmd_frame = cam.get_frame()
                cv2.imwrite(_r3.FRAME_PATH, cmd_frame.rgb)
                print("[command] VLM으로 명령 해석 중...")
                cmd = parse_pick_place(_r3.FRAME_PATH, utterance, history=dialog, holding=holding_desc)
                log_dialog(f"사용자: {utterance}")
                pick_desc, place_desc = cmd.get("pick"), cmd.get("place")
                print(f"[command] pick={pick_desc!r} place={place_desc!r} answer={cmd.get('answer')!r}")
                if cmd.get("answer"):
                    # not a manipulation command -- a question about the scene,
                    # answered from the same top-down frame, no motion
                    if speech_mod is not None:
                        say(cmd["answer"])  # speak() prints "[로봇] ..." itself
                    else:
                        print(f"[로봇] {cmd['answer']}")
                    log_dialog(f"로봇: {cmd['answer']}")
                    continue
                if pick_desc is None and place_desc is None:
                    print("명령을 이해하지 못했습니다. 다시 입력해주세요.")
                    say("명령을 이해하지 못했어요. 다시 말씀해주세요.")
                    continue
                if pick_desc is None and not holding:
                    print("잡고 있는 물체가 없어서 놓을 수 없습니다. 먼저 잡을 물체를 지정해주세요.")
                    say("지금 잡고 있는 물체가 없어요. 먼저 잡을 물체를 말씀해주세요.")
                    continue
                if pick_desc is not None and holding:
                    if place_desc is not None:
                        # already holding -- the parser sometimes hallucinates a
                        # pick target on place-only commands; just place
                        print("이미 물체를 잡고 있으므로 잡기는 건너뛰고 놓기만 수행합니다")
                        pick_desc = None
                    else:
                        print("이미 물체를 잡고 있습니다. 먼저 '(숫자)에 놔'로 내려놓은 뒤 새로 잡아주세요.")
                        say("이미 물체를 잡고 있어요. 먼저 어디에 놓을지 말씀해주세요.")
                        continue

                # launch the VLM grounding NOW, in the background, against the
                # frame the command was parsed on -- it runs while the arm
                # moves to the start pose and the TTS speaks, instead of after
                if pick_desc is not None:
                    if place_desc is not None:
                        ground_future = executor.submit(locate_many, _r3.FRAME_PATH, [pick_desc, place_desc])
                    else:
                        ground_future = executor.submit(locate, _r3.FRAME_PATH, pick_desc)
                say("네, 알겠습니다.")

            print("moving to start pose")
            hud.phase = "move to start pose"
            if holding:
                arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, args.grip_close_rad))
                show_status(cam, arm, hud, 1.0)
            else:
                arm.move_to(start_pose)
                show_status(cam, arm, hud, 1.0)
                hud.phase = "open gripper"
                arm.open_gripper()
                wait_gripper_settled_hud(arm, cam, hud, timeout=2.0)

            do_pick = args.place_mode != "command" or pick_desc is not None
            if do_pick:
                detector = "vlm" if args.place_mode == "command" else args.detector

                # ---- Stage 1: rough approach via the Record3D top-down camera ----
                print("[record3d] locating target (rough approach)")
                hud.phase = "detect target (top-down)"
                if ground_future is not None:
                    # grounding was launched right after parsing and has been
                    # running while the arm moved -- just collect the result
                    # (boxes live on the frame the command was parsed against)
                    frame = cmd_frame
                    if place_desc is not None:
                        pick_det, place_det = ground_future.result()
                    else:
                        pick_det, place_det = ground_future.result(), None
                    print(f"  VLM: pick={pick_det.answer!r} box={pick_det.box}"
                          + (f"  place={place_det.answer!r} box={place_det.box}" if place_det is not None else ""))
                    box = normalized_box_to_pixel(pick_det.box, frame.rgb.shape) if pick_det.box else None
                    if place_det is not None and place_det.box is not None:
                        pb = normalized_box_to_pixel(place_det.box, frame.rgb.shape)
                        pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                        pwx, pwy = pixel_to_world(H, pcx, pcy)
                        cmd_place_pose = Pose(x=clamp(pwx, "x"), y=clamp(pwy, "y"), z=click_place_z, t=RoArm.GRIPPER_OPEN)
                        print(f"[command] place spot pre-located: pixel=({pcx:.0f},{pcy:.0f}) -> world=({pwx:.1f},{pwy:.1f})")
                else:
                    frame = cam.get_frame()  # depth_m is intentionally never used
                    box = detect_target_pixel_box(cam, frame.rgb, pick_desc, detector, args.min_area_frac, args.rotate)
                if box is None:
                    show_preview(frame.rgb, None, None, "[record3d] no target found", args.rotate)
                    if args.place_mode == "command":
                        print("물체를 찾지 못했습니다. 다른 표현으로 다시 명령해주세요.")
                        say("물체를 찾지 못했어요. 다른 표현으로 다시 말씀해주세요.")
                        log_dialog(f"로봇: '{pick_desc}'를 화면에서 찾지 못함")
                        continue
                    if cycle == 1:
                        print("failed to locate target on the Record3D camera, aborting")
                        return 1
                    print("no more objects found; stopping loop")
                    break

                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                wx, wy = pixel_to_world(H, cx, cy)
                hud.target_box = box
                hud.target_world = (wx, wy)
                print(f"[record3d] target pixel center=({cx:.0f},{cy:.0f}) -> world (x,y)=({wx:.1f},{wy:.1f})")

                approach_pose = Pose(x=clamp(wx, "x"), y=clamp(wy, "y"), z=hover_z, t=RoArm.GRIPPER_OPEN)
                print("[record3d->approach] moving above target (rough)")
                hud.phase = f"approach hover z={hover_z:.0f}"
                arm.move_to(approach_pose, speed=args.approach_speed)
                show_status(cam, arm, hud, 1.2)

                # ---- Descend straight to grab height at the homography (x, y).
                # The grab offsets are folded into the descent target, so the
                # arm moves down and sideways in one diagonal motion.
                tx = clamp(approach_pose.x + args.grab_offset_x_mm, "x")
                ty = clamp(approach_pose.y + args.grab_offset_y_mm, "y")
                print(f"[descend] to grab z={grab_z:.1f} with offset (dx={args.grab_offset_x_mm:+.0f}, dy={args.grab_offset_y_mm:+.0f})mm -> ({tx:.1f}, {ty:.1f})")
                hud.phase = f"descend to grab z={grab_z:.0f}"
                arm.move_to(Pose(tx, ty, grab_z, RoArm.GRIPPER_OPEN), speed=args.descend_speed)
                show_status(cam, arm, hud, 1.0)

                print("picking")
                pick_hud(arm, cam, hud, carry_z, args.grip_settle_sec, args.grip_close_rad)
                holding = True
                if args.place_mode == "command":
                    holding_desc = pick_desc
                    log_dialog(f"로봇: '{pick_desc}' 파지 완료 (현재 들고 있음)")

            if args.place_mode == "command":
                if place_desc is None:
                    print("놓기 명령이 없습니다: 잡은 채로 시작 자세에서 다음 명령을 기다립니다")
                    hud.phase = "holding (awaiting place command)"
                    arm.move_to(Pose(start_pose.x, start_pose.y, carry_z, args.grip_close_rad))
                    show_status(cam, arm, hud, 1.0)
                    say("잡았습니다. 어디에 놓을지 말씀해주세요.")
                    continue

                if cmd_place_pose is None:
                    # place spot wasn't pre-located with the pick (place-only
                    # command, or it wasn't visible then): move aside first so
                    # the arm doesn't occlude the numbered spot, then find it
                    # with the VLM on a fresh top-down frame
                    hud.phase = "locate place spot (VLM)"
                    arm.move_to(Pose(start_pose.x, start_pose.y, carry_z, args.grip_close_rad))
                    show_status(cam, arm, hud, 1.2)
                    frame = cam.get_frame()
                    cv2.imwrite(_r3.FRAME_PATH, frame.rgb)
                    det = locate(_r3.FRAME_PATH, place_desc)
                    print(f"  VLM: {det.answer!r}")
                    if det.box is None:
                        print("놓을 위치를 찾지 못했습니다. 잡은 채로 다음 명령을 기다립니다.")
                        say("놓을 위치를 찾지 못했어요. 다시 말씀해주세요.")
                        log_dialog(f"로봇: 놓을 위치 '{place_desc}'를 찾지 못함 (계속 들고 있음)")
                        hud.phase = "holding (place spot not found)"
                        continue
                    pb = normalized_box_to_pixel(det.box, frame.rgb.shape)
                    pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
                    pwx, pwy = pixel_to_world(H, pcx, pcy)
                    hud.target_box = pb
                    hud.target_world = (pwx, pwy)
                    print(f"[command] place pixel=({pcx:.0f},{pcy:.0f}) -> world=({pwx:.1f},{pwy:.1f})")
                    cmd_place_pose = Pose(x=clamp(pwx, "x"), y=clamp(pwy, "y"), z=click_place_z, t=RoArm.GRIPPER_OPEN)
                else:
                    # both boxes came from the single up-front VLM call: the arm
                    # is already lifted to carry height -- go place directly
                    print(f"[command] placing directly at pre-located spot: ({cmd_place_pose.x:.1f}, {cmd_place_pose.y:.1f})")
                    hud.target_box = None
                    hud.target_world = (cmd_place_pose.x, cmd_place_pose.y)
                place_at_hud(arm, cam, hud, cmd_place_pose, carry_z, args.grip_close_rad)
                holding = False
                log_dialog(f"로봇: 들고 있던 물체({holding_desc})를 '{place_desc}' 위치에 놓음")
                holding_desc = None
                say("놓았습니다.")
            elif args.place_mode == "click":
                print("공중에서 대기 중: Record3D 화면에서 놓을 위치를 클릭하세요")
                hud.phase = "waiting for place click"
                click_place_pose = pick_place_pose_from_click(cam, H, click_place_z, args.rotate)
                if click_place_pose is None:
                    print("취소됨: 물체를 든 채 시작 자세로 복귀합니다 (그립 유지)")
                    hud.phase = "return to start (holding)"
                    arm.move_to(Pose(start_pose.x, start_pose.y, carry_z, args.grip_close_rad))
                    show_status(cam, arm, hud, 1.5)
                    break
                click_place_pose = Pose(
                    x=clamp(click_place_pose.x, "x"), y=clamp(click_place_pose.y, "y"),
                    z=click_place_pose.z, t=click_place_pose.t,
                )
                print(f"placing at clicked point: x={click_place_pose.x:.1f} y={click_place_pose.y:.1f} z={click_place_pose.z:.1f}")
                hud.target_box = None
                hud.target_world = (click_place_pose.x, click_place_pose.y)
                place_at_hud(arm, cam, hud, click_place_pose, carry_z, args.grip_close_rad)
            elif place_pose is None:
                print("returning to start pose, holding the object (no place pose -> single cycle)")
                hud.phase = "return to start (holding)"
                arm.move_to(start_pose)
                show_status(cam, arm, hud, 1.5)
                break
            else:
                print("placing at taught pose")
                hud.target_box = None
                hud.target_world = (place_pose.x, place_pose.y)
                place_at_hud(arm, cam, hud, place_pose, carry_z, args.grip_close_rad)

            print("returning to start pose (gripper open)")
            hud.phase = "return to start"
            arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
            show_status(cam, arm, hud, 1.0)

            if args.max_cycles and cycle >= args.max_cycles:
                print(f"reached max cycles ({args.max_cycles}); stopping")
                break
    except KeyboardInterrupt:
        print("\ninterrupted, stopping")
    finally:
        cam.close()
        cv2.destroyWindow(R3_WINDOW_NAME)
        if audio is not None and audio.active:
            wav = audio.stop()
            if wav is not None:
                mux_audio(cam.path, cam.video_t0, wav, audio.t0)
                Path(wav).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
