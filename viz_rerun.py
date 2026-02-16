"""Rerun-based visualizer for the comma controls challenge.

Supports multi-controller comparison, 2D trajectory rendering,
and native timeline scrubbing.

Usage:
  python viz_rerun.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --controller pid
  python viz_rerun.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --controller pid zero
  python viz_rerun.py --model_path ./models/tinyphysics.onnx --data_path ./data/ --controller pid --num_segs 5
"""

import argparse
import importlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from tinyphysics import (
    CONTROL_START_IDX, DEL_T,
    TinyPhysicsModel, TinyPhysicsSimulator, get_available_controllers,
)


def run_sim(data_path, controller_type, model_path):
    """Run a single simulation rollout and return all history arrays."""
    model = TinyPhysicsModel(model_path, debug=False)
    controller = importlib.import_module(f"controllers.{controller_type}").Controller()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    return {
        "targets": np.array(sim.target_lataccel_history),
        "actuals": np.array(sim.current_lataccel_history),
        "actions": np.array(sim.action_history),
        "v_egos": np.array([s.v_ego for s in sim.state_history]),
        "cost": cost,
    }


def integrate_trajectory(lataccels, v_egos):
    """Integrate lateral accelerations into a 2D path."""
    xs, ys = [0.0], [0.0]
    heading = 0.0
    for i in range(len(lataccels) - 1):
        v = max(v_egos[i], 1.0)
        heading += (lataccels[i] / v) * DEL_T
        xs.append(xs[-1] + v * DEL_T * math.cos(heading))
        ys.append(ys[-1] + v * DEL_T * math.sin(heading))
    return np.array(xs), np.array(ys)


def error_color(err, max_err=2.0):
    """Map tracking error to green->yellow->red color."""
    t = min(abs(err) / max_err, 1.0)
    if t < 0.5:
        s = t * 2
        return (int(46 + (241 - 46) * s), int(204 + (196 - 204) * s), int(113 + (15 - 113) * s), 255)
    s = (t - 0.5) * 2
    return (int(241 + (231 - 241) * s), int(196 + (76 - 196) * s), int(15 + (60 - 15) * s), 255)


def log_single(ctrl_name, data, seg_name=None):
    """Log one simulation result to Rerun.

    Entity tree (type-first so each view maps to a subtree):
      /trajectory/{ctrl}/target     — LineStrips2D (static)
      /trajectory/{ctrl}/actual     — LineStrips2D (static, color-coded)
      /trajectory/{ctrl}/position   — Points2D (per step)
      /lataccel/{ctrl}/target       — Scalars (per step)
      /lataccel/{ctrl}/actual       — Scalars (per step)
      /steering/{ctrl}              — Scalars (per step)
      /jerk/{ctrl}                  — Scalars (per step)
      /velocity/{ctrl}              — Scalars (per step)
      /error/{ctrl}                 — Scalars (per step)
    """
    tag = f"{ctrl_name}/{seg_name}" if seg_name else ctrl_name
    targets = data["targets"]
    actuals = data["actuals"]
    actions = data["actions"]
    v_egos = data["v_egos"]
    n_steps = len(targets)

    # Compute derived signals
    jerk = np.zeros(n_steps)
    jerk[1:] = np.diff(actuals) / DEL_T
    error = targets - actuals

    # Integrate 2D trajectories
    target_xs, target_ys = integrate_trajectory(targets, v_egos)
    actual_xs, actual_ys = integrate_trajectory(actuals, v_egos)

    # ── Static trajectory paths (always visible) ──
    target_pts = np.column_stack([target_xs, target_ys])
    rr.log(f"/trajectory/{tag}/target", rr.LineStrips2D([target_pts], colors=[(90, 95, 120, 180)]), static=True)

    segments = [np.column_stack([actual_xs[j:j+2], actual_ys[j:j+2]]) for j in range(n_steps - 1)]
    seg_colors = [error_color(error[j]) for j in range(n_steps - 1)]
    rr.log(f"/trajectory/{tag}/actual", rr.LineStrips2D(segments, colors=seg_colors), static=True)

    # ── Per-step data ──
    for i in range(n_steps):
        rr.set_time("step", sequence=i)
        rr.log(f"/lataccel/{tag}/target", rr.Scalars(targets[i]))
        rr.log(f"/lataccel/{tag}/actual", rr.Scalars(actuals[i]))
        if i < len(actions):
            rr.log(f"/steering/{tag}", rr.Scalars(actions[i]))
        rr.log(f"/jerk/{tag}", rr.Scalars(jerk[i]))
        rr.log(f"/velocity/{tag}", rr.Scalars(v_egos[i]))
        rr.log(f"/error/{tag}", rr.Scalars(error[i]))
        rr.log(
            f"/trajectory/{tag}/position",
            rr.Points2D([[actual_xs[i], actual_ys[i]]], radii=[5.0], colors=[(26, 188, 156, 255)]),
        )


def build_blueprint():
    """Build a Rerun blueprint. Each view uses origin=/<type> to auto-include all children."""
    traj_view = rrb.Spatial2DView(name="Trajectory", origin="/trajectory")
    ts_views = [
        rrb.TimeSeriesView(name=name, origin=origin)
        for name, origin in [
            ("Lat Accel", "/lataccel"),
            ("Steering", "/steering"),
            ("Jerk", "/jerk"),
            ("Velocity", "/velocity"),
            ("Error", "/error"),
        ]
    ]
    return rrb.Blueprint(
        rrb.Horizontal(
            traj_view,
            rrb.Vertical(*ts_views),
            column_shares=[2, 3],
        ),
        rrb.TimePanel(timeline="step", expanded=True),
    )


def main():
    available = get_available_controllers()
    parser = argparse.ArgumentParser(description="Rerun visualizer for comma controls challenge")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--controller", nargs="+", default=["pid"], choices=available)
    parser.add_argument("--num_segs", type=int, default=10)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    batch_mode = data_path.is_dir()

    # Save to .rrd file, then open with rerun CLI for a clean viewer
    rrd_path = "/tmp/comma_controls_viz.rrd"

    blueprint = build_blueprint()
    rr.init("comma_controls_viz", spawn=False, default_blueprint=blueprint)
    rr.save(rrd_path)
    print(f"Recording to {rrd_path}")

    if batch_mode:
        files = sorted(data_path.iterdir())[:args.num_segs]
        for ctrl in args.controller:
            print(f"Running {ctrl} on {len(files)} segments...")
            for f in files:
                result = run_sim(f, ctrl, args.model_path)
                log_single(ctrl, result, seg_name=f.stem)
                print(f"  {ctrl}/{f.stem}: total_cost={result['cost']['total_cost']:.4f}")
    else:
        for ctrl in args.controller:
            print(f"Running {ctrl} on {data_path.name}...")
            result = run_sim(data_path, ctrl, args.model_path)
            log_single(ctrl, result)
            cost = result["cost"]
            print(f"  {ctrl}: lataccel={cost['lataccel_cost']:.4f} jerk={cost['jerk_cost']:.4f} total={cost['total_cost']:.4f}")

    # Kill any stale rerun viewers to avoid confusion
    subprocess.run(["pkill", "-f", "rerun"], capture_output=True)

    import time
    time.sleep(0.5)

    print(f"Opening {rrd_path} in Rerun viewer...")
    subprocess.Popen(
        [sys.executable, "-m", "rerun", rrd_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    main()
