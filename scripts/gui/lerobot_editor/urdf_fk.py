from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LEFT_ARM_JOINTS = [f"zarm_l{i}_joint" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"zarm_r{i}_joint" for i in range(1, 8)]
LEFT_EEF_FRAME = "zarm_l7_end_effector_joint"
RIGHT_EEF_FRAME = "zarm_r7_end_effector_joint"


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def _parse_vec(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in value.split()], dtype=np.float64)


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(item) for item in rpy]
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    return np.asarray(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def transform_from_origin(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_to_rot(rpy)
    transform[:3, 3] = xyz
    return transform


def rotation_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = axis_angle_to_rot(axis, angle)
    return transform


class SimpleArmFk:
    """Small URDF FK reader for the Kuavo left/right 7-DoF arm chains.

    It intentionally supports only the joint types needed by the biped_s45 arm
    chain so the editor can run without Pinocchio in the local data environment.
    """

    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(self.urdf_path)
        self.joints = self._load_joints()
        self.left_chain = self._chain(LEFT_ARM_JOINTS, LEFT_EEF_FRAME)
        self.right_chain = self._chain(RIGHT_ARM_JOINTS, RIGHT_EEF_FRAME)

    def _load_joints(self) -> dict[str, JointSpec]:
        root = ET.parse(self.urdf_path).getroot()
        joints: dict[str, JointSpec] = {}
        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            origin = joint.find("origin")
            axis = joint.find("axis")
            joints[name] = JointSpec(
                name=name,
                joint_type=joint.attrib.get("type", "fixed"),
                xyz=_parse_vec(origin.attrib.get("xyz") if origin is not None else None, (0, 0, 0)),
                rpy=_parse_vec(origin.attrib.get("rpy") if origin is not None else None, (0, 0, 0)),
                axis=_parse_vec(axis.attrib.get("xyz") if axis is not None else None, (0, 0, 0)),
            )
        return joints

    def _chain(self, arm_joints: list[str], eef_joint: str) -> list[JointSpec]:
        names = [*arm_joints, eef_joint]
        missing = [name for name in names if name not in self.joints]
        if missing:
            raise ValueError(f"URDF is missing required joints: {missing}")
        return [self.joints[name] for name in names]

    def _fk(self, chain: list[JointSpec], q: np.ndarray) -> np.ndarray:
        if q.shape[0] != 7:
            raise ValueError(f"expected 7 arm joints, got {q.shape[0]}")
        transform = np.eye(4, dtype=np.float64)
        revolute_idx = 0
        for joint in chain:
            transform = transform @ transform_from_origin(joint.xyz, joint.rpy)
            if joint.joint_type in {"revolute", "continuous"}:
                transform = transform @ rotation_transform(joint.axis, float(q[revolute_idx]))
                revolute_idx += 1
            elif joint.joint_type != "fixed":
                raise ValueError(f"unsupported joint type {joint.joint_type!r} for {joint.name}")
        return transform[:3, 3].copy()

    def left_position(self, q_left: np.ndarray) -> np.ndarray:
        return self._fk(self.left_chain, np.asarray(q_left, dtype=np.float64).reshape(-1))

    def right_position(self, q_right: np.ndarray) -> np.ndarray:
        return self._fk(self.right_chain, np.asarray(q_right, dtype=np.float64).reshape(-1))

    def state_positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(state, dtype=np.float64).reshape(-1)
        if values.shape[0] < 15:
            raise ValueError("state/action vector must contain left and right 7-DoF arm joints")
        return self.left_position(values[:7]), self.right_position(values[8:15])

    def max_eef_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        left_a, right_a = self.state_positions(a)
        left_b, right_b = self.state_positions(b)
        return float(max(np.linalg.norm(left_b - left_a), np.linalg.norm(right_b - right_a)))

