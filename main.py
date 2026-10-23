import argparse
import math
import signal
import time
import jax
import visualize
import numpy as np
import rerun as rr

from types import FrameType
from sim.vehicle import dynamics
from sim.vehicle.dynamics import (
    VehicleParams, step_rk4, motor_torque,
    X, Y, PSI, VX, VY, OMEGA, slip_angle,
)
from sim.vehicle.state import derive_vehicle_state
from sim.comms import SimComms
from sim.scene import load_scene, MOUNT_FWD, SENSOR_Z
from sim.lidar import LidarSensor
from sim import render
from sim.odom import NoisyOdometry, OdomNoise, compose
from sim.odom_gui import OdomNoiseWindow
from sim.proto import load_can, load_proto

LIDAR_HZ: float = 10.0
ODOM_CORRECTION_HZ: float = 10.0  # perfect map -> odom correction, standing in for SLAM
VIS_HZ: float = 60.0
VELOCITY_HZ: float = 50.0
CONES_HZ: float = 1.0  # static ground truth map
CAN_TAG: int = 268  # must match the HT_CAN release drivebrain is built against

def parse_args():
    p: argparse.ArgumentParser = argparse.ArgumentParser()
    p.add_argument("--track", default="FSG", choices=["FSG", "FSI", "skidpad", "acceleration", "thin"])
    p.add_argument("--logless", action="store_true", help="Disable Rerun and the odom noise window (headless)")
    p.add_argument("--odom-seed", type=int, default=0, help="RNG seed for odometry noise")
    return p.parse_args()

def main():
    args = parse_args()
    params: VehicleParams = VehicleParams()
    dt: float = 0.004

    vis_rate: int = max(1, round(1.0 / (VIS_HZ * dt)))
    lidar_rate: int = max(1, round(1.0 / (LIDAR_HZ * dt)))
    velocity_rate: int = max(1, round(1.0 / (VELOCITY_HZ * dt)))
    cones_rate: int = max(1, round(1.0 / (CONES_HZ * dt)))
    correction_rate: int = max(1, round(1.0 / (ODOM_CORRECTION_HZ * dt)))

    s0 = dynamics.state()
    u0 = dynamics.input()
    step_rk4(s0, u0, params, dt).block_until_ready()

    scene = load_scene(args.track)
    lidar: LidarSensor = LidarSensor(scene)
    proto = load_proto()
    can = load_can(CAN_TAG)

    cone_xy: np.ndarray = np.asarray(scene.cone_xy)
    cone_type: np.ndarray = np.asarray(scene.cone_type)

    x0, y0, psi0 = [float(v) for v in scene.start_pose]
    state = dynamics.state(x=x0, y=y0, psi=psi0)

    if not args.logless:
        rr.init("vehicle_sim", spawn=True)
        render.setup_static(scene)
        visualize.setup_series()
        rr.send_blueprint(visualize.blueprint())

    odom: NoisyOdometry = NoisyOdometry(OdomNoise(), seed=args.odom_seed)
    noise_window: OdomNoiseWindow | None = None if args.logless else OdomNoiseWindow(odom.noise)
    map_from_odom = odom.map_from_odom(x0, y0, psi0)

    comms: SimComms = SimComms(proto, can)
    running: bool = True

    def stop(_sig: int, _frame: FrameType | None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    sim_t: float = 0.0
    step_i: int = 0
    wall_start: float = time.perf_counter()
    latest_pts: np.ndarray = np.empty((0, 3), dtype=np.float32)

    while running:
        comms.drain_commands()
        cmd = comms.current_input()

        u: jax.Array = dynamics.input(
            rpm_fl=cmd.rpm_fl, rpm_fr=cmd.rpm_fr,
            rpm_rl=cmd.rpm_rl, rpm_rr=cmd.rpm_rr,
            lim_fl=cmd.torque_lim_fl, lim_fr=cmd.torque_lim_fr,
            lim_rl=cmd.torque_lim_rl, lim_rr=cmd.torque_lim_rr,
            steer=cmd.wheel_steer_rad,
        )

        state: jax.Array = step_rk4(state, u, params, dt)
        sim_t += dt
        step_i += 1

        x_i: float = float(state[X])
        y_i: float = float(state[Y])
        psi_i: float = float(state[PSI])

        comms.send_velocity(float(state[VX]), float(state[VY]))
        comms.send_pose(x_i, y_i, psi_i)
        comms.send_transform(
            "map", "lidar",
            x_i + MOUNT_FWD * math.cos(psi_i),
            y_i + MOUNT_FWD * math.sin(psi_i),
            SENSOR_Z,
            psi_i,
        )

        odom.step(float(state[VX]), float(state[VY]), psi_i, float(state[OMEGA]), dt)
        if (step_i - 1) % correction_rate == 0:
            map_from_odom = odom.map_from_odom(x_i, y_i, psi_i)
            comms.send_transform("map", "odom", map_from_odom[0], map_from_odom[1], 0.0, map_from_odom[2])
        comms.send_transform("odom", "base_link", odom.x, odom.y, 0.0, odom.psi)

        # comms.send_state(derive_vehicle_state(state, u, params))  # raw packed struct replaced by typed GT messages below

        if step_i % lidar_rate == 0:
            latest_pts = lidar.scan((x_i, y_i, psi_i))
            comms.send_pointcloud(latest_pts)
            if not args.logless:
                rr.set_time("sim_time", duration=sim_t)
                render.log_points(latest_pts)

        if step_i % cones_rate == 0:
            comms.send_cones(cone_xy, cone_type)

        if step_i % vis_rate == 0 and not args.logless:
            rr.set_time("sim_time", duration=sim_t)
            render.log_car(x_i, y_i, psi_i, comms.timed_out())
            render.log_odom_car(*compose(map_from_odom, odom.pose))
            noise_window.poll()
            tau_fl, tau_fr, tau_rl, tau_rr = (float(t) for t in motor_torque(state, u, params))
            visualize.log_scalars(
                speed=math.hypot(float(state[VX]), float(state[VY])),
                yaw_rate=float(state[OMEGA]),
                slip=float(slip_angle(state)),
                torque_fl=tau_fl, torque_fr=tau_fr,
                torque_rl=tau_rl, torque_rr=tau_rr,
                steering=cmd.wheel_steer_rad,
            )

        drift: float = (wall_start + sim_t) - time.perf_counter()
        if drift > 0:
            time.sleep(drift)

    comms.close()

if __name__ == "__main__":
    main()
