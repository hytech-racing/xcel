"""Simulated wheel/IMU odometry: integrates noisy body-frame velocity and yaw rate into an
odom -> base_link pose, and computes the exact map -> odom correction a perfect SLAM would publish."""
import math
from dataclasses import dataclass

import numpy as np


Pose2 = tuple[float, float, float]  # x, y, yaw


def compose(a: Pose2, b: Pose2) -> Pose2:
    """a * b for 2D poses, e.g. map_from_odom * odom_from_base = map_from_base."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]


@dataclass
class OdomNoise:
    vel_std: float = 0.05       # m/s, white noise per sim step on each body velocity axis
    yaw_rate_std: float = 0.01  # rad/s, white noise per sim step
    vel_scale_err: float = 0.0  # fraction, e.g. 0.02 = speed reads 2% high
    gyro_bias: float = 0.0      # rad/s, constant yaw rate offset


class NoisyOdometry:
    def __init__(self, noise: OdomNoise, seed: int = 0) -> None:
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        # odom starts where the car is switched on, like a real estimator
        self.x: float = 0.0
        self.y: float = 0.0
        self.psi: float = 0.0

    def step(self, vx_world: float, vy_world: float, psi_world: float, yaw_rate: float, dt: float) -> None:
        """Integrate one sim step of noisy measurements taken from the ground-truth state."""
        c, s = math.cos(psi_world), math.sin(psi_world)
        vx_body = c * vx_world + s * vy_world
        vy_body = -s * vx_world + c * vy_world

        n = self.noise
        vel_noise = self._rng.normal(0.0, n.vel_std, 2) if n.vel_std > 0 else (0.0, 0.0)
        yaw_noise = self._rng.normal(0.0, n.yaw_rate_std) if n.yaw_rate_std > 0 else 0.0
        vx_meas = vx_body * (1.0 + n.vel_scale_err) + vel_noise[0]
        vy_meas = vy_body * (1.0 + n.vel_scale_err) + vel_noise[1]
        yaw_rate_meas = yaw_rate + n.gyro_bias + yaw_noise

        heading = self.psi + 0.5 * yaw_rate_meas * dt  # midpoint, so zero noise means zero drift
        c, s = math.cos(heading), math.sin(heading)
        self.x += (c * vx_meas - s * vy_meas) * dt
        self.y += (s * vx_meas + c * vy_meas) * dt
        self.psi += yaw_rate_meas * dt

    @property
    def pose(self) -> Pose2:
        return self.x, self.y, self.psi

    def map_from_odom(self, x_map: float, y_map: float, psi_map: float) -> Pose2:
        """map -> odom such that map -> odom -> base_link lands exactly on the ground-truth pose."""
        yaw = psi_map - self.psi
        c, s = math.cos(yaw), math.sin(yaw)
        return x_map - (c * self.x - s * self.y), y_map - (s * self.x + c * self.y), yaw
