"""RoArm-M2 pick-and-place driven ENTIRELY by a trained end-to-end
Diffusion Policy: start pose -> rough approach -> fine positioning ->
descent -> grab, all as one learned policy. No homography, no VLM, no
tracker anywhere in this file -- see pick_place_fixed_to_eih.py for the
classical/hybrid version this replaces (which still uses homography for
stage 1). Placing is still classical (taught pose), matching the scope of
the demonstrations dp_collect_data_e2e.py records (start pose through grab
only).

Both cameras are used as observations: the fixed overview camera and the
eye-in-hand camera, exactly as recorded by dp_collect_data_e2e.py.

Requires a model trained with dp_train_e2e.py on episodes collected with
dp_collect_data_e2e.py -- there is no fallback detector here; this script
does nothing useful without a trained checkpoint.

Run:
  python3 pick_place_e2e_dp.py --fixed-camera-index 1 --camera-index 0 \\
      --dp-checkpoint dp_e2e_checkpoints/dp_e2e_latest.pt --dp-stats dp_e2e_checkpoints/stats.json \\
      --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import argparse
import os

import cv2

from dp_policy_runner_e2e import DiffusionPolicyE2ERunner
from pick_place import (
    WORKSPACE,
    clamp,
    expand_workspace,
    load_place_pose,
    load_start_pose,
    load_workspace_calibration,
    RecordingCapture,
    pick as eih_pick,
    place_at as eih_place_at,
    sleep_showing as eih_sleep_showing,
    wait_gripper_settled as eih_wait_gripper_settled,
)
from roarm_driver import Pose, RoArm

FIXED_WINDOW_NAME = "RoArm E2E-DP Fixed Camera"
EIH_WINDOW_NAME = "RoArm E2E-DP Eye-in-Hand"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RoArm-M2 pick-and-place driven entirely by an end-to-end Diffusion Policy (no homography/VLM)")
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "/dev/cu.usbserial-120"))
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--start-pose", default="roarm_start_pose.json")
    p.add_argument("--fixed-camera-index", type=int, default=1)
    p.add_argument("--camera-index", type=int, default=0, help="eye-in-hand camera index")
    p.add_argument("--record", dest="record", action="store_true", default=True)
    p.add_argument("--no-record", dest="record", action="store_false")

    p.add_argument("--dp-checkpoint", default="dp_e2e_checkpoints/dp_e2e_latest.pt")
    p.add_argument("--dp-stats", default="dp_e2e_checkpoints/stats.json")
    p.add_argument("--dp-exec-horizon", type=int, default=6, help="how many predicted actions to execute before re-observing and re-predicting")
    p.add_argument("--dp-inference-steps", type=int, default=16)
    p.add_argument("--dp-max-steps", type=int, default=150, help="give up (return failure) if the policy hasn't closed the gripper within this many executed actions")
    p.add_argument("--approach-speed", type=float, default=0.3, help="arm speed (0-1) for policy-commanded moves")

    p.add_argument("--place-pose", default="place_pose.json")
    p.add_argument("--place-x", type=float, default=None)
    p.add_argument("--place-y", type=float, default=None)
    p.add_argument("--place-z", type=float, default=None)
    p.add_argument("--carry-z", type=float, default=None)
    p.add_argument("--grip-settle-sec", type=float, default=0.5)
    p.add_argument("--grip-close-rad", type=float, default=3.25)
    p.add_argument("--floor-calibration", default="floor_calibration.json")
    p.add_argument("--workspace-calibration", default="workspace_calibration.json")
    p.add_argument("--max-cycles", type=int, default=0)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

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

    expand_workspace(start_pose)
    if place_pose is not None:
        expand_workspace(place_pose)
    measured_workspace = load_workspace_calibration(args.workspace_calibration)
    if measured_workspace is not None:
        WORKSPACE["x"] = measured_workspace["x"]
        WORKSPACE["y"] = measured_workspace["y"]
    print(f"workspace: x={WORKSPACE['x']}, y={WORKSPACE['y']}, z={WORKSPACE['z']}")

    carry_z = clamp(args.carry_z if args.carry_z is not None else start_pose.z, "z")
    print(f"carry height (transit z): {carry_z:.1f}")

    fixed_cap = cv2.VideoCapture(args.fixed_camera_index)
    if not fixed_cap.isOpened():
        raise RuntimeError(f"cannot open fixed camera index {args.fixed_camera_index}")
    cv2.namedWindow(FIXED_WINDOW_NAME, cv2.WINDOW_NORMAL)

    eih_cap = RecordingCapture(args.camera_index) if args.record else cv2.VideoCapture(args.camera_index)
    if not eih_cap.isOpened():
        raise RuntimeError(f"cannot open eye-in-hand camera index {args.camera_index}")
    if args.record and getattr(eih_cap, "path", None):
        print(f"eye-in-hand 카메라 영상 녹화 중: {eih_cap.path}")
    cv2.namedWindow(EIH_WINDOW_NAME, cv2.WINDOW_NORMAL)

    print(f"[DP-E2E] loading checkpoint={args.dp_checkpoint} stats={args.dp_stats}")
    dp_runner = DiffusionPolicyE2ERunner(
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

            print("[DP-E2E] running the full approach+grab policy")
            ok = dp_runner.run(arm, fixed_cap, eih_cap, show_window=EIH_WINDOW_NAME)
            if not ok:
                if cycle == 1:
                    print("policy failed to grab, aborting")
                    return 1
                print("no more objects found; stopping loop")
                break

            print("picking (settle/lift)")
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
