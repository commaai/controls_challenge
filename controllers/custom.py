from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  PID controller with future-plan lookahead (delay compensation).

  The simulated car responds to a steer command with roughly 1-2 steps
  (0.1-0.2 s) of lag, so tracking the instantaneous target always leaves
  the controller a step behind. Instead, the error is computed against a
  blend of the current target and the next few steps of the future plan:
  the controller starts steering for a curve before the curve arrives.
  Averaging over the near future also low-pass filters the target, which
  reduces jerk.

  Gains were tuned by grid search and validated on held-out segments
  (see the module docstring above / solution.txt).
  """

  def __init__(self, p=0.18, i=0.13, d=-0.04, lookahead=4, ff=0.0,
               integral_clamp=10.0, action_smooth=0.0):
    self.p = p
    self.i = i
    self.d = d
    self.lookahead = lookahead          # future-plan steps blended into the target
    self.ff = ff                        # gain on anticipated target change
    self.integral_clamp = integral_clamp
    self.action_smooth = action_smooth  # 0 = no smoothing, else low-pass factor
    self.error_integral = 0.0
    self.prev_error = None
    self.prev_action = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    fp = future_plan.lataccel
    n = min(self.lookahead, len(fp))

    # Delay-compensated target: mean of current target and next n planned targets
    if n > 0:
      target = float(np.mean([target_lataccel] + list(fp[:n])))
    else:
      target = target_lataccel

    error = target - current_lataccel
    self.error_integral = np.clip(self.error_integral + error,
                                  -self.integral_clamp, self.integral_clamp)
    error_diff = 0.0 if self.prev_error is None else error - self.prev_error
    self.prev_error = error

    action = (self.p * error
              + self.i * self.error_integral
              + self.d * error_diff
              + self.ff * (target - target_lataccel))

    if self.action_smooth > 0:
      action = (1 - self.action_smooth) * action + self.action_smooth * self.prev_action
    self.prev_action = action
    return action
