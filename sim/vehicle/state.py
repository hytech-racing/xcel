import struct
from dataclasses import dataclass

import jax
from .dynamics import VehicleParams, xdot, X, Y, PSI, VX, VY, OMEGA

_INPUT_FMT: str = "9d"
_STATE_FMT: str = "9d"

INPUT_SIZE: int = struct.calcsize(_INPUT_FMT)
STATE_SIZE: int = struct.calcsize(_STATE_FMT)

@dataclass
class VehicleInput:
    rpm_fl: float = 0.0
    rpm_fr: float = 0.0
    rpm_rl: float = 0.0
    rpm_rr: float = 0.0
    torque_lim_fl: float = 0.0
    torque_lim_fr: float = 0.0
    torque_lim_rl: float = 0.0
    torque_lim_rr: float = 0.0
    wheel_steer_rad: float = 0.0

    def pack(self) -> bytes:
        return struct.pack(
            _INPUT_FMT,
            self.rpm_fl, self.rpm_fr,
            self.rpm_rl, self.rpm_rr,
            self.torque_lim_fl, self.torque_lim_fr,
            self.torque_lim_rl, self.torque_lim_rr,
            self.wheel_steer_rad,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "VehicleInput":
        return cls(*struct.unpack(_INPUT_FMT, data))

@dataclass
class VehicleState:
    ax_world: float = 0.0
    ay_world: float = 0.0
    psi_ddot_world: float = 0.0
    vx_world: float = 0.0
    vy_world: float = 0.0
    psi_dot_world: float = 0.0
    x_world: float = 0.0
    y_world: float = 0.0
    psi_world: float = 0.0

    def pack(self) -> bytes:
        return struct.pack(
            _STATE_FMT,
            self.ax_world, self.ay_world, self.psi_ddot_world,
            self.vx_world, self.vy_world, self.psi_dot_world,
            self.x_world, self.y_world, self.psi_world,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "VehicleState":
        return cls(*struct.unpack(_STATE_FMT, data))

def derive_vehicle_state(state: jax.Array, input: jax.Array, params: VehicleParams) -> VehicleState:
    """Derives VehicleState from the JAX state and input"""
    deriv: jax.Array = xdot(state, input, params)
    return VehicleState(
        ax_world=float(deriv[3]),
        ay_world=float(deriv[4]),
        psi_ddot_world=float(deriv[5]),
        vx_world=float(state[VX]),
        vy_world=float(state[VY]),
        psi_dot_world=float(state[OMEGA]),
        x_world=float(state[X]),
        y_world=float(state[Y]),
        psi_world=float(state[PSI]),
    )
