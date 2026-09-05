# all of this is slop
import math

import numpy as np
import mujoco
import rerun as rr
import rerun.blueprint as rrb

from sim.world import World, CAR_Z

CAR_HALF_L: float = 1.2
CAR_HALF_W: float = 0.6
CAR_HALF_H: float = 0.2
CONE_RADIUS: float = 0.114
CONE_HALF_H: float = 0.1625
LIDAR_FWD: float = 0.75
LIDAR_UP: float = 0.15

def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="world"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="torques [Nm]", contents=["scalars/torque_*"]),
                rrb.TimeSeriesView(name="steering [rad]", contents=["scalars/steering"]),
                rrb.TimeSeriesView(name="speed [m/s]", contents=["scalars/speed"]),
                rrb.TimeSeriesView(name="yaw rate [r/s]", contents=["scalars/yaw_rate"]),
                rrb.TimeSeriesView(name="slip [rad]", contents=["scalars/slip"]),
            ),
            column_shares=[2, 1],
        )
    )

def setup_static(world: World) -> None:
    if world.cones_left:
        pts: list[list[float]] = [[c[0], c[1], CONE_HALF_H] for c in world.cones_left]
        rr.log("world/cones/blue", rr.Points3D(pts, radii=CONE_RADIUS,
               colors=[[30, 100, 255]] * len(pts)), static=True)
    if world.cones_right:
        pts = [[c[0], c[1], CONE_HALF_H] for c in world.cones_right]
        rr.log("world/cones/yellow", rr.Points3D(pts, radii=CONE_RADIUS,
               colors=[[255, 210, 0]] * len(pts)), static=True)
    rr.log("scalars/torque_fl", rr.SeriesLines(colors=[[255, 80, 80]], names=["FL"]), static=True)
    rr.log("scalars/torque_fr", rr.SeriesLines(colors=[[80, 255, 80]], names=["FR"]), static=True)
    rr.log("scalars/torque_rl", rr.SeriesLines(colors=[[80, 80, 255]], names=["RL"]), static=True)
    rr.log("scalars/torque_rr", rr.SeriesLines(colors=[[255, 200, 0]], names=["RR"]), static=True)
    rr.log("scalars/steering", rr.SeriesLines(colors=[[200, 200, 200]], names=["steer"]), static=True)
    rr.log("scalars/speed", rr.SeriesLines(colors=[[0, 220, 220]], names=["speed"]), static=True)
    rr.log("scalars/yaw_rate", rr.SeriesLines(colors=[[220, 100, 220]], names=["yaw_rate"]), static=True)
    rr.log("scalars/slip", rr.SeriesLines(colors=[[255, 140, 0]], names=["slip"]), static=True)

def _planner_arc_points(arc, count: int = 64) -> np.ndarray:
    begin = np.array([arc.begin.x_map_m, arc.begin.y_map_m], dtype=np.float64)
    end = np.array([arc.end.x_map_m, arc.end.y_map_m], dtype=np.float64)
    chord = end - begin
    chord_length = float(np.linalg.norm(chord))
    curvature = float(arc.signed_curvature_inv_m)
    if chord_length < 1e-9:
        return np.empty((0, 3), dtype=np.float32)
    if abs(curvature) < 1e-9:
        xy = np.linspace(begin, end, count)
    else:
        radius = 1.0 / abs(curvature)
        half_chord = 0.5 * chord_length
        if half_chord > radius:
            xy = np.linspace(begin, end, count)
        else:
            midpoint = 0.5 * (begin + end)
            left_normal = np.array([-chord[1], chord[0]]) / chord_length
            center = midpoint + math.copysign(
                math.sqrt(max(0.0, radius * radius - half_chord * half_chord)),
                curvature,
            ) * left_normal
            start_angle = math.atan2(begin[1] - center[1], begin[0] - center[0])
            end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
            delta = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
            angles = start_angle + np.linspace(0.0, delta, count)
            xy = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return np.column_stack((xy, np.full(len(xy), 0.08))).astype(np.float32)

def _circle_points(x: float, y: float, radius: float, count: int = 96) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=True)
    return np.column_stack((
        x + radius * np.cos(angles),
        y + radius * np.sin(angles),
        np.full(count, 0.06),
    )).astype(np.float32)

def _add_mujoco_sphere(scene, point: np.ndarray, radius: float, color) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]), np.asarray(point, dtype=np.float64),
        np.eye(3).reshape(-1), np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1

def _add_mujoco_line(scene, begin: np.ndarray, end: np.ndarray, width: float, color) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
        np.eye(3).reshape(-1), np.asarray(color, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_LINE, width,
        np.asarray(begin, dtype=np.float64), np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1

def draw_mujoco_planner(viewer, planner_visualization, x: float, y: float) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    if planner_visualization is None:
        return

    midpoints = np.array(
        [[point.x_map_m, point.y_map_m, 0.12]
         for point in planner_visualization.midpoints], dtype=np.float64,
    )
    for point in midpoints:
        _add_mujoco_sphere(scene, point, 0.09, [1.0, 0.1, 1.0, 1.0])
    for begin, end in zip(midpoints[:-1], midpoints[1:]):
        _add_mujoco_line(scene, begin, end, 4.0, [1.0, 0.1, 1.0, 1.0])

    arc = _planner_arc_points(planner_visualization.planner_arc)
    for begin, end in zip(arc[:-1], arc[1:]):
        _add_mujoco_line(scene, begin, end, 5.0, [0.1, 0.9, 1.0, 1.0])

    lookahead = float(planner_visualization.lookahead_distance_m)
    if lookahead > 0.0:
        circle = _circle_points(x, y, lookahead)
        for begin, end in zip(circle[:-1], circle[1:]):
            _add_mujoco_line(scene, begin, end, 3.0, [1.0, 1.0, 1.0, 1.0])

def log_frame(
    *,
    sim_t: float,
    x: float, y: float, psi: float,
    pts_local: np.ndarray,
    speed: float,
    yaw_rate: float,
    slip: float,
    torque_fl: float, torque_fr: float,
    torque_rl: float, torque_rr: float,
    steering: float,
    cmd_timed_out: bool,
) -> None:
    rr.set_time("sim_time", duration=sim_t)
    hpsi: float = psi * 0.5
    rr.log("world/car", rr.Boxes3D(
        centers=[[x, y, CAR_Z]],
        half_sizes=[[CAR_HALF_L, CAR_HALF_W, CAR_HALF_H]],
        quaternions=[rr.Quaternion(xyzw=[0.0, 0.0, math.sin(hpsi), math.cos(hpsi)])],
        colors=[[220, 50, 50] if not cmd_timed_out else [100, 100, 100]],
    ))
    if pts_local.shape[0] > 0:
        cp: float = math.cos(psi)
        sp: float = math.sin(psi)
        R_z: np.ndarray = np.array([
            [cp, -sp, 0.0],
            [sp, cp, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        sensor_pos: np.ndarray = np.array([
            x + LIDAR_FWD * cp,
            y + LIDAR_FWD * sp,
            CAR_Z + LIDAR_UP,
        ], dtype=np.float32)
        pts_world: np.ndarray = pts_local @ R_z.T + sensor_pos
        rr.log("world/lidar", rr.Points3D(pts_world, radii=0.05,
               colors=[[0, 255, 120]] * len(pts_world)))
    rr.log("scalars/torque_fl", rr.Scalars(torque_fl))
    rr.log("scalars/torque_fr", rr.Scalars(torque_fr))
    rr.log("scalars/torque_rl", rr.Scalars(torque_rl))
    rr.log("scalars/torque_rr", rr.Scalars(torque_rr))
    rr.log("scalars/steering", rr.Scalars(steering))
    rr.log("scalars/speed", rr.Scalars(speed))
    rr.log("scalars/yaw_rate", rr.Scalars(yaw_rate))
    rr.log("scalars/slip", rr.Scalars(slip))
