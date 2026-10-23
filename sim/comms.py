import math
import time
import types

import numpy as np
import zmq

import os

from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud

from .vehicle import VehicleInput, VehicleState, INPUT_SIZE
from .vehicle.dynamics import MAX_RPM

SIM_BIND_HOST: str = os.environ.get("SIM_BIND_HOST", "127.0.0.1")

def _endpoint(port: int) -> str:
    return f"tcp://{SIM_BIND_HOST}:{port}"

ZMQ_PREFIX: str = "tcp:///tmp/drivebrain_sim_"
SEND_SOCKET_PORT: int = 6767
RECV_SOCKET_PORT: int = 5940
LIDAR_SOCKET_PORT: int = 1155

INPUT_TIMEOUT: float = 1.0 # secs

POINT_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
POINT_STRIDE: int = POINT_DTYPE.itemsize  # 12

# Steering calibration, mirrored from drivebrain's estimation::SteeringEstimator.
# drivebrain maps wheel angle -> analog -> column angle; we invert that chain.
STEER_ANALOG: np.ndarray = np.array([1164.0, 2059.0, 2960.0])
STEER_WHEEL_DEG: np.ndarray = np.array([20.11, 0.0, -20.86])
COLUMN_OFFSET: float = 2092.0
COLUMN_SCALING: float = -0.09641

def column_deg_to_wheel_rad(column_deg: float) -> float:
    """Steering column angle (deg) -> front wheel angle (rad). Clamps at the table ends like drivebrain"""
    analog: float = column_deg / COLUMN_SCALING + COLUMN_OFFSET
    return math.radians(float(np.interp(analog, STEER_ANALOG, STEER_WHEEL_DEG)))

class SimComms:
    def __init__(self, proto: types.ModuleType, can: types.ModuleType) -> None:
        self._proto = proto
        self._can = can
        self._ctx: zmq.Context = zmq.Context()

        # NOTE: socket options must be set BEFORE bind() to apply to every pipe a
        # peer opens; setting them after leaves an early-connecting consumer on the
        # 1000-deep default queue, which is exactly how a backlog builds.
        self._cmd: zmq.Socket = self._ctx.socket(zmq.PULL)
        self._cmd.setsockopt(zmq.RCVHWM, 1)
        self._cmd.bind(_endpoint(RECV_SOCKET_PORT))

        # multipart channel. carries all low-rate ground-truth data (pose, cones, transforms).
        # multipart rules out CONFLATE, so keep the queue shallow to bound transform staleness.
        # velocity, pose, map->lidar and odom->base_link go out every step (plus map->odom at 10 Hz)
        # in one burst, before the I/O thread drains any of it; a limit near the burst size drops
        # whatever is sent last. 20 = about 4 steps of backlog.
        self._state: zmq.Socket = self._ctx.socket(zmq.PUSH)
        self._state.setsockopt(zmq.SNDHWM, 20)
        self._state.setsockopt(zmq.LINGER, 0)
        self._state.bind(_endpoint(SEND_SOCKET_PORT))

        # lidar is single-part and only-latest-matters -> CONFLATE keeps just the newest
        # scan queued, so a slow Foxglove/bridge can never accumulate stale clouds.
        self._lidar: zmq.Socket = self._ctx.socket(zmq.PUSH)
        self._lidar.setsockopt(zmq.CONFLATE, 1)
        self._lidar.setsockopt(zmq.LINGER, 0)
        self._lidar.bind(_endpoint(LIDAR_SOCKET_PORT))

        self._latest_input: VehicleInput = VehicleInput()
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
        inp: VehicleInput = self._latest_input
        if type_name == "hytech.drivebrain_speed_set_input":
            msg = self._can.hytech_pb2.drivebrain_speed_set_input()
            msg.ParseFromString(body)
            inp.rpm_fl = msg.drivebrain_set_rpm_fl
            inp.rpm_fr = msg.drivebrain_set_rpm_fr
            inp.rpm_rl = msg.drivebrain_set_rpm_rl
            inp.rpm_rr = msg.drivebrain_set_rpm_rr
        elif type_name == "hytech.drivebrain_torque_lim_input":
            msg = self._can.hytech_pb2.drivebrain_torque_lim_input()
            msg.ParseFromString(body)
            inp.torque_lim_fl = msg.drivebrain_torque_fl
            inp.torque_lim_fr = msg.drivebrain_torque_fr
            inp.torque_lim_rl = msg.drivebrain_torque_rl
            inp.torque_lim_rr = msg.drivebrain_torque_rr
        elif type_name == "hytech.drivebrain_desired_torque_input":
            # legacy torque mode (e.g. teleop): express it as a saturated speed request
            msg = self._can.hytech_pb2.drivebrain_desired_torque_input()
            msg.ParseFromString(body)
            torques = (msg.drivebrain_torque_fl, msg.drivebrain_torque_fr,
                       msg.drivebrain_torque_rl, msg.drivebrain_torque_rr)
            inp.rpm_fl, inp.rpm_fr, inp.rpm_rl, inp.rpm_rr = (math.copysign(MAX_RPM, t) for t in torques)
            inp.torque_lim_fl, inp.torque_lim_fr, inp.torque_lim_rl, inp.torque_lim_rr = (abs(t) for t in torques)
        elif type_name == "hytech.drivebrain_steering_input":
            msg = self._can.hytech_pb2.drivebrain_steering_input()
            msg.ParseFromString(body)
            inp.wheel_steer_rad = column_deg_to_wheel_rad(msg.drivebrain_steering)
        else:
            return

        self._last_cmd_time = time.monotonic()

    def timed_out(self) -> bool:
        return time.monotonic() - self._last_cmd_time > INPUT_TIMEOUT

    def current_input(self) -> VehicleInput:
        """Return the latest input, zeroing torque limits (coasting) if the controller timed out."""
        if self.timed_out():
            return VehicleInput(wheel_steer_rad=self._latest_input.wheel_steer_rad)
        return self._latest_input

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

    def send_velocity(self, vx_world: float, vy_world: float) -> None:
        """Send ground-truth map-frame velocity to drivebrain's VNData input."""
        msg = self._proto.hytech_msgs_pb2.VNData()
        msg.vn_vel_m_s.x = vx_world
        msg.vn_vel_m_s.y = vy_world
        msg.vn_vel_m_s.z = 0.0
        self._send_typed(msg)

    def send_pointcloud(self, pts: np.ndarray) -> None:
        n: int = len(pts)
        buf = np.empty(n, dtype=POINT_DTYPE)
        buf['x'] = pts[:, 0]
        buf['y'] = pts[:, 1]
        buf['z'] = pts[:, 2]

        PEF = PackedElementField
        msg = PointCloud(
            frame_id="lidar",
            point_stride=POINT_STRIDE,
            fields=[PEF(name="x", offset=0, type=PEF.FLOAT32),
                    PEF(name="y", offset=4, type=PEF.FLOAT32),
                    PEF(name="z", offset=8, type=PEF.FLOAT32),
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

    def send_cones(self, cone_xy: np.ndarray, cone_type: np.ndarray) -> None:
        """Ground-truth track map. cone_type maps 1:1 to dv_msgs Cones.ConeColor."""
        msg = self._proto.dv_msgs_pb2.Cones(timestamp_us=int(time.time() * 1e6))
        for (x, y), c in zip(cone_xy, cone_type):
            cone = msg.cones.add()
            cone.position.x = float(x)
            cone.position.y = float(y)
            cone.color = int(c)
        self._send_typed(msg)

    def send_transform(self, parent: str, child: str, x: float, y: float, z: float, psi: float) -> None:
        """parent -> child transform, yaw-only rotation"""
        msg = FrameTransform(parent_frame_id=parent, child_frame_id=child)
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
