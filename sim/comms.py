import math
import time
import types

import numpy as np
import zmq

from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud

from .vehicle import VehicleInput, VehicleState, INPUT_SIZE

ZMQ_PREFIX: str = "ipc:///tmp/drivebrain_sim_"
SEND_SOCKET_PORT: int = 6767
RECV_SOCKET_PORT: int = 5940
LIDAR_SOCKET_PORT: int = 1155

INPUT_TIMEOUT: float = 1.0 # secs

POINT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1'),
])
POINT_STRIDE: int = POINT_DTYPE.itemsize  # 16

class SimComms:
    def __init__(self, proto: types.ModuleType, can: types.ModuleType) -> None:
        self._proto = proto
        self._can = can
        self._ctx: zmq.Context = zmq.Context()

        self._cmd: zmq.Socket = self._ctx.socket(zmq.PULL)
        self._cmd.bind(f"{ZMQ_PREFIX}{RECV_SOCKET_PORT}")
        self._cmd.setsockopt(zmq.RCVHWM, 1)

        # multipart channel. carries all low-rate ground-truth data (pose, cones, transforms)
        self._state: zmq.Socket = self._ctx.socket(zmq.PUSH)
        self._state.bind(f"{ZMQ_PREFIX}{SEND_SOCKET_PORT}")
        self._state.setsockopt(zmq.SNDHWM, 10)

        self._lidar: zmq.Socket = self._ctx.socket(zmq.PUSH)

        self._lidar.bind(f"{ZMQ_PREFIX}{LIDAR_SOCKET_PORT}")
        self._lidar.setsockopt(zmq.SNDHWM, 1)

        self._latest_input: VehicleInput = VehicleInput()
        self._planner_visualization = None
        self._last_cmd_time: float = time.monotonic()

    def drain_commands(self) -> None:
        while True:
            try:
                parts: list[bytes] = self._cmd.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) < 2:
                continue
            self._handle_command(parts[0].decode(), parts[1])

    def _handle_command(self, type_name: str, body: bytes) -> None:
        """Unpack a drivebrain CAN command into the latest vehicle input."""
        if type_name == "hytech.drivebrain_desired_torque_input":
            msg = self._can.hytech_pb2.drivebrain_desired_torque_input()
            msg.ParseFromString(body)
            self._latest_input.torque_fl = msg.drivebrain_torque_fl
            self._latest_input.torque_fr = msg.drivebrain_torque_fr
            self._latest_input.torque_rl = msg.drivebrain_torque_rl
            self._latest_input.torque_rr = msg.drivebrain_torque_rr
        elif type_name == "hytech.drivebrain_steering_input":
            msg = self._can.hytech_pb2.drivebrain_steering_input()
            msg.ParseFromString(body)
            self._latest_input.wheel_steer_rad = math.radians(msg.drivebrain_steering)
        elif type_name == "hytech_msgs.PlannerVisualization":
            msg = self._proto.autonomy_msgs_pb2.PlannerVisualization()
            msg.ParseFromString(body)
            self._planner_visualization = msg
            return
        else:
            return

        self._last_cmd_time = time.monotonic()

    def timed_out(self) -> bool:
        return time.monotonic() - self._last_cmd_time > INPUT_TIMEOUT

    def current_input(self) -> VehicleInput:
        """Return the latest input, zeroing torques if the controller timed out."""
        if self.timed_out():
            return VehicleInput(wheel_steer_rad=self._latest_input.wheel_steer_rad)
        return self._latest_input

    def planner_visualization(self):
        return self._planner_visualization

    def _send_typed(self, msg) -> None:
        """Multipart (type name, body) frame"""
        try:
            self._state.send_multipart(
                [msg.DESCRIPTOR.full_name.encode(), msg.SerializeToString()], zmq.NOBLOCK
            )
        except zmq.Again:
            pass

    def send_pose(self, x: float, y: float, psi: float) -> None:
        """Ground-truth vehicle pose as hytech_msgs.pose, map frame."""
        msg = self._proto.base_msgs_pb2.pose()
        msg.position.x = x
        msg.position.y = y
        msg.orientation.z = math.sin(psi * 0.5)
        msg.orientation.w = math.cos(psi * 0.5)
        self._send_typed(msg)

    def send_pointcloud(self, pts: np.ndarray, rgba: np.ndarray) -> None:
        n: int = len(pts)
        buf = np.empty(n, dtype=POINT_DTYPE)
        buf['x'] = pts[:, 0]
        buf['y'] = pts[:, 1]
        buf['z'] = pts[:, 2]
        buf['r'] = rgba[:, 0]
        buf['g'] = rgba[:, 1]
        buf['b'] = rgba[:, 2]
        buf['a'] = rgba[:, 3]

        PEF = PackedElementField
        msg = PointCloud(
            frame_id="lidar",
            point_stride=POINT_STRIDE,
            fields=[PEF(name="x", offset=0,  type=PEF.FLOAT32),
                    PEF(name="y", offset=4,  type=PEF.FLOAT32),
                    PEF(name="z", offset=8,  type=PEF.FLOAT32),
                    PEF(name="red", offset=12, type=PEF.UINT8),
                    PEF(name="green", offset=13, type=PEF.UINT8),
                    PEF(name="blue", offset=14, type=PEF.UINT8),
                    PEF(name="alpha", offset=15, type=PEF.UINT8),
            ],
            data=buf.tobytes(),
        )
        # capture time — consumers derive scan age from this, and slam needs it to pair
        # a scan with odometry. wall clock, to match how drivebrain stamps its logs.
        msg.timestamp.GetCurrentTime()
        try:
            self._lidar.send(msg.SerializeToString(), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def send_cones(self, cones_left: list[list[float]], cones_right: list[list[float]]) -> None:
        """Ground-truth track map. left = blue, right = yellow"""
        msg = self._proto.dv_msgs_pb2.Cones(timestamp_us=int(time.time() * 1e6))
        for xy in cones_left:
            cone = msg.cones.add()
            cone.position.x = xy[0]
            cone.position.y = xy[1]
            cone.color = self._proto.dv_msgs_pb2.Cones.BLUE
        for xy in cones_right:
            cone = msg.cones.add()
            cone.position.x = xy[0]
            cone.position.y = xy[1]
            cone.color = self._proto.dv_msgs_pb2.Cones.YELLOW
        self._send_typed(msg)

    def send_transform(self, x: float, y: float, z: float, psi: float) -> None:
        """map -> lidar transform: pose of the LIDAR SITE (car pose + mount offset), not the car origin."""
        msg = FrameTransform(parent_frame_id="map", child_frame_id="lidar")
        msg.timestamp.GetCurrentTime()
        msg.translation.x = x
        msg.translation.y = y
        msg.translation.z = z
        msg.rotation.z = math.sin(psi * 0.5)
        msg.rotation.w = math.cos(psi * 0.5)
        self._send_typed(msg)

    def close(self) -> None:
        self._cmd.close()
        self._state.close(linger=0)
        self._lidar.close()
        self._ctx.term()
