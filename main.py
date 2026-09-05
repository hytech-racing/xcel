import argparse
import math
import signal
import time
import jax
import mujoco
import mujoco.viewer
import visualize
import numpy as np
import rerun as rr

from types import FrameType
from sim.vehicle import dynamics
from sim.vehicle.dynamics import (
    VehicleParams, step_rk4,
    X, Y, PSI, VX, VY, OMEGA, slip_angle,
)
from sim.vehicle.state import derive_vehicle_state
from sim.comms import SimComms
from sim.world import World
from sim.lidar import LidarSensor
from sim.proto import load_can, load_proto

VIEWER_HZ: float = 60.0
LIDAR_HZ: float = 20.0
CONES_HZ: float = 1.0  # static ground truth map
CAN_TAG: int = 268  # must match the HT_CAN release drivebrain is built against

def parse_args():
    p: argparse.ArgumentParser = argparse.ArgumentParser()
    p.add_argument("--track", default="FSG", choices=["FSG", "FSI", "skidpad", "acceleration", "thin"])
    p.add_argument("--logless", action="store_true", help="Disable Rerun time-series")
    p.add_argument("--headless", action="store_true", help="Disable MuJoCo viewer")
    return p.parse_args()

def main():
    """
    """
    args = parse_args()
    params: VehicleParams = VehicleParams()
    dt: float = 0.002

    vis_rate: int = max(1, round(1.0 / (VIEWER_HZ * dt)))
    lidar_rate: int = max(1, round(1.0 / (LIDAR_HZ * dt)))
    cones_rate: int = max(1, round(1.0 / (CONES_HZ * dt)))

    s0 = dynamics.state()
    u0 = dynamics.input()
    step_rk4(s0, u0, params, dt).block_until_ready()

    world: World = World(args.track)
    lidar: LidarSensor = LidarSensor(world)
    proto = load_proto()
    can = load_can(CAN_TAG)

    x0: float
    y0: float
    psi0: float
    x0, y0, psi0 = [float(v) for v in world.start_pose]
    state = dynamics.state(x=x0, y=y0, psi=psi0)
    world.set_car_pose(x0, y0, psi0)

    if not args.logless:
        rr.init("vehicle_sim", spawn=True)
        visualize.setup_static(world)
        rr.send_blueprint(visualize.blueprint())

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(world.model, world.data)
        viewer.cam.lookat[:] = [x0, y0, 0.5]
        viewer.cam.distance = 20.0
        viewer.cam.elevation = -25.0

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

    while running and (viewer is None or viewer.is_running()):
        comms.drain_commands()
        cmd = comms.current_input()

        u: jax.Array = dynamics.input(
            tau_fl=cmd.torque_fl, tau_fr=cmd.torque_fr,
            tau_rl=cmd.torque_rl, tau_rr=cmd.torque_rr,
            steer=cmd.wheel_steer_rad,
        )

        state: jax.Array = step_rk4(state, u, params, dt)
        sim_t += dt
        step_i += 1

        x_i: float = float(state[X])
        y_i: float = float(state[Y])
        psi_i: float = float(state[PSI])

        world.set_car_pose(x_i, y_i, psi_i)
        # comms.send_state(derive_vehicle_state(state, u, params))  # raw packed struct replaced by typed GT messages below

        if step_i % lidar_rate == 0:
            latest_pts, latest_rgba = lidar.scan()
            comms.send_pointcloud(latest_pts, latest_rgba)
            comms.send_pose(x_i, y_i, psi_i)
            # transform carries the lidar SITE pose — same mount math as visualize.log_frame
            comms.send_transform(
                x_i + visualize.LIDAR_FWD * math.cos(psi_i),
                y_i + visualize.LIDAR_FWD * math.sin(psi_i),
                visualize.CAR_Z + visualize.LIDAR_UP,
                psi_i,
            )

        if step_i % cones_rate == 0:
            comms.send_cones(world.cones_left, world.cones_right)

        if step_i % vis_rate == 0:
            if viewer is not None:
                visualize.draw_mujoco_planner(
                    viewer, comms.planner_visualization(), x_i, y_i
                )
                viewer.sync()
            if not args.logless:
                visualize.log_frame(
                    sim_t=sim_t,
                    x=x_i, y=y_i, psi=psi_i,
                    pts_local=latest_pts,
                    speed=math.hypot(float(state[VX]), float(state[VY])),
                    yaw_rate=float(state[OMEGA]),
                    slip=float(slip_angle(state)),
                    torque_fl=cmd.torque_fl, torque_fr=cmd.torque_fr,
                    torque_rl=cmd.torque_rl, torque_rr=cmd.torque_rr,
                    steering=cmd.wheel_steer_rad,
                    cmd_timed_out=comms.timed_out(),
                )

        drift: float = (wall_start + sim_t) - time.perf_counter()
        if drift > 0:
            time.sleep(drift)

    comms.close()

if __name__ == "__main__":
    main()
