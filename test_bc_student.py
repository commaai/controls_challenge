from __future__ import annotations

import numpy as np

from tinyphysics import FuturePlan, State


def make_short_plan() -> FuturePlan:
  return FuturePlan(
    lataccel=[0.1, 0.2, 0.3],
    roll_lataccel=[0.0, 0.01, 0.02],
    v_ego=[22.0, 22.1, 22.2],
    a_ego=[0.0, 0.01, 0.0],
  )


def test_bc_student_loads_artifact_and_returns_finite_action():
  from controllers.bc_student import MODEL_PATH, Controller

  assert MODEL_PATH.exists()

  controller = Controller()
  action = controller.update(
    target_lataccel=0.15,
    current_lataccel=0.05,
    state=State(roll_lataccel=0.01, v_ego=22.0, a_ego=0.0),
    future_plan=make_short_plan(),
  )

  assert np.isfinite(action)
  assert -2.0 <= action <= 2.0


def test_bc_student_reuses_loaded_model():
  from controllers.bc_student import Controller

  state = State(roll_lataccel=0.01, v_ego=22.0, a_ego=0.0)
  plan = make_short_plan()
  first = Controller()
  second = Controller()

  first.update(0.15, 0.05, state, plan)
  second.update(0.15, 0.05, state, plan)

  assert first.model is second.model
