"""RoArm-M2 low-level driver.

JSON command set per official Waveshare docs (waveshareteam/roarm_m2, json_cmd.h):
  T=100  CMD_MOVE_INIT           {"T":100}
  T=104  CMD_XYZT_GOAL_CTRL      {"T":104,"x":..,"y":..,"z":..,"t":..,"spd":..}   (interpolated)
  T=1041 CMD_XYZT_DIRECT_CTRL    {"T":1041,"x":..,"y":..,"z":..,"t":..}           (direct, for servoing)
  T=105  CMD_SERVO_RAD_FEEDBACK  {"T":105} -> {x,y,z,t,torB,torS,torE}
  T=106  CMD_EOAT_HAND_CTRL      {"T":106,"cmd":<rad>,"spd":..,"acc":..} (1.57=open .. 3.14=closed)
  T=121  CMD_SINGLE_JOINT_ANGLE  {"T":121,"joint":1-4,"angle":deg,"spd":..,"acc":..}
  T=122  CMD_JOINTS_ANGLE_CTRL   {"T":122,"b":..,"s":..,"e":..,"h":..,"spd":..,"acc":..}
  T=210  CMD_TORQUE_CTRL         {"T":210,"cmd":0|1}

Transport per official examples:
  HTTP:   GET http://<ip>/js?json=<url-encoded json>
  Serial: write json line + b"\n" @ 115200 baud
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import requests


@dataclass
class Pose:
    x: float
    y: float
    z: float
    t: float

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z, "t": self.t}


class RoArm:
    GRIPPER_OPEN = 1.57
    GRIPPER_CLOSED = 3.14
    GRIPPER_MAX = 3.4  # firm-close ceiling, a bit past nominal "closed" for a tighter grip

    def __init__(self, ip: str | None = None, serial_port: str | None = None, baudrate: int = 115200):
        self.ip = ip
        self.serial_port = serial_port
        self.baudrate = baudrate
        self._serial = None

    # ---- transport -------------------------------------------------
    def _send_serial(self, payload: dict) -> dict | None:
        import serial

        if self._serial is None or not self._serial.is_open:
            self._serial = serial.Serial(self.serial_port, baudrate=self.baudrate, timeout=1.0, dsrdtr=None)
            self._serial.setRTS(False)
            self._serial.setDTR(False)

        self._serial.reset_input_buffer()
        self._serial.write(json.dumps(payload).encode("utf-8") + b"\n")
        self._serial.flush()

        if payload.get("T") != 105:
            return {"sent": True}

        deadline = time.time() + 1.0
        last = None
        while time.time() < deadline:
            line = self._serial.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(k in last for k in ("x", "y", "z")):
                return last
        return last

    def _send_http(self, payload: dict) -> dict | None:
        r = requests.get(f"http://{self.ip}/js", params={"json": json.dumps(payload)}, timeout=1.0)
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            return None

    def send(self, payload: dict) -> dict | None:
        if self.serial_port:
            return self._send_serial(payload)
        return self._send_http(payload)

    # ---- commands ----------------------------------------------------
    def move_init(self) -> None:
        self.send({"T": 100})

    def feedback(self) -> Pose | None:
        res = self.send({"T": 105})
        if not res or not all(k in res for k in ("x", "y", "z")):
            return None
        return Pose(x=float(res["x"]), y=float(res["y"]), z=float(res["z"]), t=float(res.get("t", self.GRIPPER_CLOSED)))

    def move_to(self, pose: Pose, speed: float = 0.25) -> None:
        """Interpolated move (T:104) -- use for larger repositioning moves."""
        self.send({"T": 104, "x": pose.x, "y": pose.y, "z": pose.z, "t": pose.t, "spd": speed})

    def servo_to(self, pose: Pose) -> None:
        """Direct move (T:1041) -- use for small steps in a visual-servo loop."""
        self.send({"T": 1041, "x": pose.x, "y": pose.y, "z": pose.z, "t": pose.t})

    def set_gripper(self, rad: float, speed: float = 0, acc: float = 0) -> None:
        rad = max(self.GRIPPER_OPEN, min(self.GRIPPER_MAX, rad))
        self.send({"T": 106, "cmd": rad, "spd": speed, "acc": acc})

    def open_gripper(self) -> None:
        self.set_gripper(self.GRIPPER_OPEN)

    def close_gripper(self) -> None:
        self.set_gripper(self.GRIPPER_CLOSED)

    def torque(self, on: bool) -> None:
        self.send({"T": 210, "cmd": 1 if on else 0})
