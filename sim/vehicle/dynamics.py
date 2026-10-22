"""
JAX Two-Track Vehicle Dynamics Model

Reimplementation of the OpenModelica PlanarMechanics two-track model
with SlipBasedWheelJoint tire model.

State vector (10 x 1):
[x, y, psi, vx, vy, omega, omega_fl, omega_fr, omega_rl, omega_rr]

x, y: CG position in world frame (m)
psi: Heading angle (rad)
vx, vy: CG velocity in world frame (m/s)
omega: Yaw rate (rad/s)
omega_fl..rr: Wheel angular velocities (rad/s)

Input vector (9 x 1):
[rpm_fl, rpm_fr, rpm_rl, rpm_rr, lim_fl, lim_fr, lim_rl, lim_rr, delta]

rpm_fl..rr: Motor speed setpoints (rpm)
lim_fl..rr: Motor torque limits (Nm, sign ignored)
delta: Front wheel steering angle (rad)

The inverters run in speed mode like on the car: each motor's torque is a P speed
loop clipped to +/- its torque limit. Requesting MAX_RPM (or 0 / -MAX_RPM) saturates
the loop, so the limit becomes the applied torque -- the same trick drivebrain uses
to get torque control through speed mode.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

class VehicleParams(NamedTuple):
    m: float = 250.0
    I_zz: float = 120.0
    l_f: float = 0.765
    l_r: float = 0.765
    w_f: float = 0.6
    w_r: float = 0.6
    r_w: float = 0.2
    I_w: float = 0.094
    gear_ratio: float = 11.83
    f_n: float = 600.0
    mu_a: float = 1.8
    mu_s: float = 1.5
    s_adh: float = 0.117
    s_sld: float = 0.35
    v_adh_min: float = 0.005
    v_sld_min: float = 0.02
    c_rr: float = 0.015
    w_rr_eps: float = 0.1
    speed_kp: float = 0.02  # inverter speed loop gain (Nm/rpm)

X, Y, PSI, VX, VY, OMEGA, W_FL, W_FR, W_RL, W_RR = range(10)
"""Numerical indicies for the vehicle state JAX array"""

RPM_FL, RPM_FR, RPM_RL, RPM_RR, LIM_FL, LIM_FR, LIM_RL, LIM_RR, DELTA = range(9)
"""Numerical indicies for the vehicle input JAX array"""

MAX_RPM: float = 20000.0
"""Speed setpoint drivebrain sends to saturate the speed loop into pure torque control"""

RAD_S_TO_RPM: float = 60.0 / (2.0 * jnp.pi)

def _tire_friction(
    v_slip: jax.Array,
    v_adh: jax.Array,
    v_sld: jax.Array,
    mu_a: float,
    mu_s: float,
    f_n: float,
) -> jax.Array:
    """Friction force magnitude from slip velocity magnitude"""
    eps: float = 1e-8
    f_adhesion = mu_a * f_n * v_slip / jnp.maximum(v_adh, eps)
    t = jnp.clip((v_slip - v_adh) / jnp.maximum(v_sld - v_adh, eps), 0.0, 1.0)
    mu_t = mu_a - (mu_a - mu_s) * 0.5 * (1.0 - jnp.cos(jnp.pi * t))
    f_transition = mu_t * f_n
    f_sliding = mu_s * f_n
    return jnp.where(v_slip < v_adh, f_adhesion,
                     jnp.where(v_slip < v_sld, f_transition, f_sliding))

def state(
    x: float = 0., y: float = 0., psi: float = 0.,
    vx: float = 0., vy: float = 0., omega: float = 0.,
    w_fl: float = 0., w_fr: float = 0.,
    w_rl: float = 0., w_rr: float = 0.,
) -> jax.Array:
    """Returns a JAX array of a vehicle state"""
    return jnp.array([x, y, psi, vx, vy, omega, w_fl, w_fr, w_rl, w_rr])

def input(
    rpm_fl: float = 0., rpm_fr: float = 0.,
    rpm_rl: float = 0., rpm_rr: float = 0.,
    lim_fl: float = 0., lim_fr: float = 0.,
    lim_rl: float = 0., lim_rr: float = 0.,
    steer: float = 0.,
) -> jax.Array:
    """Returns a JAX array of a vehicle input"""
    return jnp.array([rpm_fl, rpm_fr, rpm_rl, rpm_rr, lim_fl, lim_fr, lim_rl, lim_rr, steer])

def motor_torque(state: jax.Array, u: jax.Array, params: VehicleParams) -> jax.Array:
    """Motor torques (Nm) from the inverter speed loop, clipped to the torque limits"""
    rpm: jax.Array = state[W_FL:W_RR + 1] * params.gear_ratio * RAD_S_TO_RPM
    lim: jax.Array = jnp.abs(u[LIM_FL:LIM_RR + 1])
    return jnp.clip(params.speed_kp * (u[RPM_FL:RPM_RR + 1] - rpm), -lim, lim)

@jax.jit
def xdot(state: jax.Array, u: jax.Array, params: VehicleParams) -> jax.Array:
    """State derivative x_dot = f(x, u)."""
    p: VehicleParams = params
    eps: float = 1e-10

    psi: jax.Array = state[PSI]
    vx: jax.Array = state[VX]
    vy: jax.Array = state[VY]
    omega: jax.Array = state[OMEGA]
    w: jax.Array = state[6:10]

    tau_drive: jax.Array = motor_torque(state, u, p) * p.gear_ratio
    delta: jax.Array = u[DELTA]

    c_psi: jax.Array = jnp.cos(psi)
    s_psi: jax.Array = jnp.sin(psi)

    r_body: jax.Array = jnp.array([
        [ p.l_f,  p.w_f],
        [ p.l_f, -p.w_f],
        [-p.l_r,  p.w_r],
        [-p.l_r, -p.w_r],
    ])

    r_wx: jax.Array = c_psi * r_body[:, 0] - s_psi * r_body[:, 1]
    r_wy: jax.Array = s_psi * r_body[:, 0] + c_psi * r_body[:, 1]

    v_cx: jax.Array = vx - omega * r_wy
    v_cy: jax.Array = vy + omega * r_wx

    steer: jax.Array = jnp.array([delta, delta, 0.0, 0.0])
    heading: jax.Array = psi + steer
    c_h: jax.Array = jnp.cos(heading)
    s_h: jax.Array = jnp.sin(heading)

    v_long: jax.Array = v_cx * c_h + v_cy * s_h
    v_lat: jax.Array = -v_cx * s_h + v_cy * c_h

    v_slip_long: jax.Array = v_long - w * p.r_w
    v_slip_lat: jax.Array = v_lat
    v_slip_mag: jax.Array = jnp.sqrt(v_slip_long**2 + v_slip_lat**2 + eps)

    v_adh: jax.Array = jnp.maximum(p.s_adh * jnp.abs(v_long), p.v_adh_min)
    v_sld: jax.Array = jnp.maximum(p.s_sld * jnp.abs(v_long), p.v_sld_min)

    f_mag: jax.Array = _tire_friction(v_slip_mag, v_adh, v_sld, p.mu_a, p.mu_s, p.f_n)
    f_long: jax.Array = -f_mag * v_slip_long / v_slip_mag
    f_lat: jax.Array = -f_mag * v_slip_lat / v_slip_mag

    Fx: jax.Array = f_long * c_h - f_lat * s_h
    Fy: jax.Array = f_long * s_h + f_lat * c_h

    ax: jax.Array = jnp.sum(Fx) / p.m
    ay: jax.Array = jnp.sum(Fy) / p.m
    alpha: jax.Array = jnp.sum(r_wx * Fy - r_wy * Fx) / p.I_zz

    # Rolling resistance, opposing wheel rotation. tanh instead of sign RK4 agrees at standstill
    tau_rr: jax.Array = -jnp.tanh(w / p.w_rr_eps) * p.c_rr * p.f_n * p.r_w

    # I_w * w_dot = tau_drive + tau_rr - f_long * r_w
    w_dot: jax.Array = (tau_drive + tau_rr - f_long * p.r_w) / p.I_w

    return jnp.array([
        vx, vy, omega,
        ax, ay, alpha,
        w_dot[0], w_dot[1],
        w_dot[2], w_dot[3],
    ])

RK4_SUBSTEPS: int = 16
"""The tire model is very stiff near standstill. Without substeps the speed loop chatters
there and a 0 rpm request stalls the car at ~0.5 m/s instead of stopping it"""

def _rk4(state: jax.Array, u: jax.Array, params: VehicleParams, dt: float) -> jax.Array:
    k1: jax.Array = xdot(state, u, params)
    k2: jax.Array = xdot(state + 0.5 * dt * k1, u, params)
    k3: jax.Array = xdot(state + 0.5 * dt * k2, u, params)
    k4: jax.Array = xdot(state + dt * k3, u, params)
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

@jax.jit
def step_rk4(state: jax.Array, u: jax.Array, params: VehicleParams, dt: float) -> jax.Array:
    """Advance dt with RK4_SUBSTEPS RK4 substeps, zero-order hold on input"""
    h: float = dt / RK4_SUBSTEPS
    return jax.lax.fori_loop(0, RK4_SUBSTEPS, lambda _, s: _rk4(s, u, params, h), state)


def slip_angle(s: jax.Array) -> jax.Array:
    psi: jax.Array = s[..., PSI]
    c: jax.Array = jnp.cos(psi)
    sin: jax.Array = jnp.sin(psi)
    vx_b: jax.Array = c * s[..., VX] + sin * s[..., VY]
    vy_b: jax.Array = -sin * s[..., VX] + c * s[..., VY]
    return jnp.arctan2(vy_b, jnp.maximum(vx_b, 1e-3))

