"""RoArm-M2 two-stage pick-and-place: fixed overview webcam for the rough
approach, eye-in-hand camera + VLM for the precise final descent. No LiDAR.

Stage 1 -- rough approach (fixed camera, e.g. an Iriun-connected phone or any
webcam that stays put, index via --fixed-camera-index):
  one VLM call locates the target on that camera's view, its pixel center is
  mapped to a real-world (x, y) via the one-time ground-plane homography
  (calibrate_ground_homography.py), and the arm moves there at a safe hover
  height. Same geometry as pick_place_fixed_cam.py, just stopping short of
  the floor instead of grabbing.

  The arm has no separate camera-tilt joint (Pose is just x/y/z + gripper
  angle), so "look down" is done the only way this hardware allows: after
  the hover move, the arm lowers straight down a bit further (still well
  above grab height, --lookdown-clearance-mm above the floor) before the
  eye-in-hand camera is asked to look. Getting physically closer is what
  gives that camera a decent close-up view of the target instead of the
  wide, small-in-frame view it would get from all the way up at hover height.

Stage 2 -- precise descent (eye-in-hand camera, mounted on the gripper,
index via --camera-index) -- --policy selects how this stage is driven:
  vlm (default) -- reuses pick_place.py's track_and_descend unchanged: ONE
    more VLM call to (re)find the target in this camera's view, then a
    local tracker (ViT-Track) follows it through every look-then-move step
    of the descent, no further API calls and no manual re-drawing.
  dp -- a Diffusion Policy trained on hand-collected demonstrations
    (dp_collect_data.py + dp_train.py) drives the descent+grab directly
    from eye-in-hand pixels instead of VLM+tracker+geometry. Needs
    --dp-checkpoint/--dp-stats pointing at a trained model; see
    dp_policy_runner.py.

With --policy vlm, the handoff from the "macro" fixed camera to the
eye-in-hand camera is itself driven by the VLM (a fresh detection in the
new view), not a blind coordinate carry-over -- exactly two VLM calls per
cycle, one per camera. With --policy dp, stage 1 still uses one VLM call
(the fixed camera's rough approach is unchanged either way) but stage 2
uses zero -- the learned policy replaces it entirely.

Run:
  python3 pick_place_fixed_to_eih.py --target "the red block" \\
      --fixed-camera-index 1 --camera-index 0 --serial-port /dev/cu.usbserial-120
  python3 pick_place_fixed_to_eih.py --detector manual --fixed-camera-index 1 --camera-index 0 ...   # no API calls at all
  python3 pick_place_fixed_to_eih.py --target "the red block" --policy dp \\
      --dp-checkpoint dp_checkpoints/dp_latest.pt --dp-stats dp_checkpoints/stats.json \\
      --fixed-camera-index 1 --camera-index 0 --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import os

import cv2

import pick_place as _pp
import pick_place_fixed_cam as _fc
from pick_place import (
    WORKSPACE,
    clamp,
    expand_workspace,
    load_camera_hfov,
    load_floor_z,
    load_place_pose,
    load_start_pose,
    load_workspace_calibration,
    RecordingCapture,
    pick as eih_pick,
    place_at as eih_place_at,
    sleep_showing as eih_sleep_showing,
    track_and_descend,
    wait_gripper_settled as eih_wait_gripper_settled,
)
from pick_place_fixed_cam import (
    GroundGridOverlay,
    detect_target_pixel_box,
    load_homography,
    pick_place_pose_from_click,
    pixel_to_world,
)
from roarm_driver import Pose, RoArm

FIXED_WINDOW_NAME = _fc.WINDOW_NAME  # "RoArm Fixed-Camera Pick-Place"
EIH_WINDOW_NAME = _pp.WINDOW_NAME  # "RoArm Eye-in-Hand"


def build_parser():
    p = _pp.build_parser()
    p.description = "RoArm-M2 pick-and-place: fixed camera for rough approach, eye-in-hand + VLM for precise descent (no LiDAR)"
    # --camera-index (from pick_place.build_parser) is the EYE-IN-HAND camera here.
    for action in p._actions:
        if action.dest == "camera_index":
            action.help = "eye-in-hand camera index (mounted on the gripper) -- drives the precise final descent"
    p.add_argument("--fixed-camera-index", type=int, default=1, help="fixed overview camera index (e.g. an Iriun-connected phone) -- drives the rough approach")
    p.add_argument("--homography", default="ground_homography.json", help="output of calibrate_ground_homography.py")
    p.add_argument("--grid-spacing-mm", type=float, default=50.0, help="AR overlay on the fixed-camera window: grid spacing in real mm (0 disables)")
    p.add_argument("--hover-clearance-mm", type=float, default=120.0, help="height above the calibrated floor to stop the rough approach at (stage 1, fixed camera)")
    p.add_argument("--lookdown-clearance-mm", type=float, default=60.0, help="height above the calibrated floor the arm lowers to (still straight down, x/y unchanged) right before the eye-in-hand camera looks -- there is no separate tilt joint, so getting physically closer is how it 'looks down' at the target")
    p.add_argument("--lookdown-speed", type=float, default=0.4, help="arm speed (0-1) for the hover -> lookdown descent")
    p.add_argument("--policy", choices=("vlm", "dp"), default="vlm", help="what drives stage 2 (eye-in-hand descend+grab): 'vlm' = track_and_descend (VLM+tracker), 'dp' = a trained Diffusion Policy (see dp_train.py)")
    p.add_argument("--dp-checkpoint", default="dp_checkpoints/dp_latest.pt", help="--policy dp: trained model checkpoint from dp_train.py")
    p.add_argument("--dp-stats", default="dp_checkpoints/stats.json", help="--policy dp: normalization stats saved alongside the checkpoint by dp_train.py")
    p.add_argument("--dp-exec-horizon", type=int, default=4, help="--policy dp: how many predicted actions to execute before re-observing and re-predicting (receding horizon)")
    p.add_argument("--dp-inference-steps", type=int, default=16, help="--policy dp: DDIM sampling steps per look (fewer = faster, usually fine)")
    p.add_argument("--dp-max-steps", type=int, default=60, help="--policy dp: give up (return failure) if the policy hasn't closed the gripper within this many executed actions")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.vlm_backend:
        os.environ["VLM_BACKEND"] = args.vlm_backend

    H, cal_camera_index = load_homography(args.homography)
    if cal_camera_index != args.fixed_camera_index:
        print(f"주의: homography는 camera-index {cal_camera_index}로 캘리브레이션됐지만 지금은 {args.fixed_camera_index}를 사용합니다.")

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)
    start_pose = load_start_pose(args.start_pose)

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

    floor_z = load_floor_z(args.floor_calibration)
    if floor_z is not None:
        print(f"floor calibration loaded: z={floor_z:.1f} (from {args.floor_calibration})")
    else:
        print(f"no floor calibration found at {args.floor_calibration}; using --pick-down-mm/--place-down-mm and --hover-clearance-mm relative to the start pose instead")

    calibrated_hfov = load_camera_hfov(args.camera_calibration)
    if calibrated_hfov is not None:
        args.camera_hfov_deg = calibrated_hfov
        print(f"camera FOV calibration loaded: hfov={calibrated_hfov:.1f} deg (from {args.camera_calibration})")

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
    print(f"rough-approach hover height (stage 1, fixed camera): {hover_z:.1f}")

    if floor_z is not None:
        lookdown_z = clamp(floor_z + args.lookdown_clearance_mm, "z")
    else:
        lookdown_z = clamp(hover_z - args.hover_clearance_mm * 0.5, "z")
    lookdown_z = min(lookdown_z, hover_z)  # never go up
    print(f"look-down height (right before eye-in-hand VLM call): {lookdown_z:.1f}")

    # Stage 1 camera: fixed overview
    fixed_cap = cv2.VideoCapture(args.fixed_camera_index)
    if not fixed_cap.isOpened():
        raise RuntimeError(f"cannot open fixed camera index {args.fixed_camera_index}")
    cv2.namedWindow(FIXED_WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.grid_spacing_mm > 0:
        _fc._ground_grid = GroundGridOverlay(args.homography, grid_spacing_mm=args.grid_spacing_mm)
        if _fc._ground_grid.enabled:
            print(f"AR 그리드 오버레이 켜짐 (간격 {args.grid_spacing_mm:.0f}mm)")

    # Stage 2 camera: eye-in-hand
    eih_cap = RecordingCapture(args.camera_index) if args.record else cv2.VideoCapture(args.camera_index)
    if not eih_cap.isOpened():
        raise RuntimeError(f"cannot open eye-in-hand camera index {args.camera_index}")
    if args.record and getattr(eih_cap, "path", None):
        print(f"eye-in-hand 카메라 영상 녹화 중: {eih_cap.path}")
    cv2.namedWindow(EIH_WINDOW_NAME, cv2.WINDOW_NORMAL)

    dp_runner = None
    if args.policy == "dp":
        # imported lazily so `--policy vlm` (the default) never needs torch installed
        from dp_policy_runner import DiffusionPolicyRunner

        print(f"[DP] loading checkpoint={args.dp_checkpoint} stats={args.dp_stats}")
        dp_runner = DiffusionPolicyRunner(
            checkpoint_path=args.dp_checkpoint,
            stats_path=args.dp_stats,
            exec_horizon=args.dp_exec_horizon,
            num_inference_steps=args.dp_inference_steps,
            grip_close_threshold=args.grip_close_rad,
            max_steps=args.dp_max_steps,
            move_speed=args.approach_speed,
        )

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

            # ---- Stage 1: rough approach via the fixed camera ----
            print("[fixed-cam] locating target (rough approach)")
            frame = _fc.capture_frame(fixed_cap)
            box = detect_target_pixel_box(fixed_cap, frame, args.target, args.detector, args.min_area_frac)
            if box is None:
                _fc.show_preview(frame, None, None, "[fixed-cam] no target found")
                if cycle == 1:
                    print("failed to locate target on the fixed camera, aborting")
                    return 1
                print("no more objects found; stopping loop")
                break

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            wx, wy = pixel_to_world(H, cx, cy)
            _fc.show_preview(frame, box, (wx, wy), f"[fixed-cam] pixel=({cx:.0f},{cy:.0f}) -> world=({wx:.1f},{wy:.1f})")
            print(f"[fixed-cam] target pixel center=({cx:.0f},{cy:.0f}) -> world (x,y)=({wx:.1f},{wy:.1f})")

            approach_pose = Pose(x=clamp(wx, "x"), y=clamp(wy, "y"), z=hover_z, t=RoArm.GRIPPER_OPEN)
            print("[fixed-cam->approach] moving above target (rough)")
            arm.move_to(approach_pose, speed=args.approach_speed)
            eih_sleep_showing(eih_cap, 1.2)

            # ---- Look down: no separate tilt joint, so lower straight down a
            # bit (same x/y) before asking the eye-in-hand camera to look --
            # this is what gets the target big/clear enough in that view.
            lookdown_pose = Pose(x=approach_pose.x, y=approach_pose.y, z=lookdown_z, t=RoArm.GRIPPER_OPEN)
            print(f"[look-down] descending to z={lookdown_z:.1f} before the eye-in-hand VLM call")
            arm.move_to(lookdown_pose, speed=args.lookdown_speed)
            eih_sleep_showing(eih_cap, 1.0)

            # ---- Stage 2: precise descent via the eye-in-hand camera ----
            print(f"[eye-in-hand] acquiring target and descending (policy={args.policy})")
            if args.policy == "dp":
                stage2_ok = dp_runner.run(arm, eih_cap, show_window=EIH_WINDOW_NAME)
            else:
                # track_and_descend makes exactly one VLM call to (re)acquire
                # the target in THIS camera's view, then a local tracker
                # follows it through the whole descent -- the VLM-driven handoff.
                stage2_ok = track_and_descend(arm, eih_cap, args.target, floor_z, args)
            if not stage2_ok:
                if cycle == 1:
                    print("failed to locate/descend to target with the eye-in-hand camera, aborting")
                    return 1
                print("no more objects found; stopping loop")
                break

            print("picking")
            eih_pick(arm, eih_cap, carry_z, args.grip_settle_sec, args.grip_close_rad)

            if place_pose is None:
                print("returning to start pose, holding the object (no place pose -> single cycle)")
                arm.move_to(start_pose)
                eih_sleep_showing(eih_cap, 1.5)
                arm.close_gripper()
                eih_sleep_showing(eih_cap, 0.3)
                break

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
        fixed_cap.release()
        cv2.destroyWindow(FIXED_WINDOW_NAME)
        eih_cap.release()
        cv2.destroyWindow(EIH_WINDOW_NAME)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
