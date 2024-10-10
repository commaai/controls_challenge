from . import BaseController
import numpy as np
from typing import List, Tuple

ACC_G = 9.81

class Controller(BaseController):
  """
  A PIDF controller with bycicle dynamic model feedforward
  """
  def __init__(self,):


    self.alpha = 0.9697
    self.lag = 5
    self.action_history = []
    self.error_history = []
    self.la_history = []
    self.last_state = None

    # Debug
    self.predict_la_history = []
    self.target_la_history = []
    self.delta_action_history = []

  def optimize_mpc(alpha: float, x_0: float, n: int, plan: List[float], W_d: List[float], W_j: List[float]) -> Tuple[List[float], float]:
    '''
    Model:
        dx = alpha * du
        x: state (lataccel)
        u: input (steer command)

    Params:
        alpha: steer to lataccel coefficient
        x_0: initial lataccel
        u_0: last control
        n: size of prediction horizon, same as control horizon
        plan: lataccel targets for the length of prediction horizon
        W_d: Weights for deviation error
        W_j: Weights for jerk error (len n-1)
    
    Return:
        control: optimal control for the control horizon
        prediction: predicted future states following the optimal control
        obj: objective value for the optimal control
    '''

    # Solve for minima of objective function using closed form solution to gradient
    upper_alpha = np.triu(alpha * np.ones((n, n)))
    lower_alpha = np.tril(alpha * np.ones((n, n)))
    W_j = np.array(W_j)
    d1 = -2 * W_j # Off diagonal of DuJ
    d0 = np.zeros(n)
    d0[: -1] += 2 * W_j
    d0[1: ] += 2 * W_j
    DuJ = np.diag(d0) + np.diag(d1, 1) + np.diag(d1, -1)
    plan = np.array(plan)
    W_d = np.array(W_d)
    A = 2 * upper_alpha @ np.diag(W_d) @ lower_alpha + DuJ
    b = 2 * upper_alpha @ np.diag(W_d) @ (plan - x_0)
    control = np.linalg.solve(A, b)
    prediction = x_0 + lower_alpha @ control

    L_d = (W_d * (prediction - plan) ** 2).sum()
    L_j = (W_j * np.diff(control, n=1) **2).sum()

    return control, prediction, L_d, L_j
  
  def update(self, target_lataccel, current_lataccel, state, future_plan):
      '''
      target_la_t
      state_t
      action_t
      action_t-lag-1 & la_t-lag-2 -> la_t-1

      la_history[-1] = la_t-1
      action_history[-1] = action_t-1
      last_state = state_t-1

      predict_la[-1] = la_t+lag-1

      '''

      horizon_n = 10
      # Update state history
      if self.action_history:
        self.la_history.append(current_lataccel - self.last_state[0])

      if len(self.action_history) < self.lag + 1 or len(future_plan[0]) < self.lag:
        action = 0
      else:
        predict_la = [self.la_history[-1]]
        for t in reversed(range(1, self.lag + 1)):
          delta_action = self.action_history[-t] - self.action_history[-t-1]
          predict_la.append(predict_la[-1] + delta_action * self.alpha)

        self.predict_la_history.append(predict_la[-1])
        target_la = future_plan[0][self.lag-1] - future_plan[1][self.lag-1] if self.lag > 0 else target_lataccel - state[0]
        self.target_la_history.append(target_la)
        target_delta_la = target_la - predict_la[-1]
        self.error_history.append(target_delta_la)
        target_delta_action = target_delta_la / self.alpha
        self.delta_action_history.append(target_delta_action)
        target_delta_action = np.clip(target_delta_action, -0.01, 0.01)
        action = self.action_history[-1] + target_delta_action
        # action *= 0.25

      self.action_history.append(action)
      self.last_state = state
      
      # # Feedforward
      # target_steer_la = target_lataccel - state[0]
      # pred_steer = self.predict_steer(target_steer_la, state[1])

      # # PID Feedback
      # error = (target_lataccel - current_lataccel)
      # self.error_integral += error
      # error_diff = error - self.prev_error
      # self.prev_error = error

      # command = self.f * pred_steer + self.p * error + self.i * self.error_integral + self.d * error_diff
      # self.steer_command_history.append(command)
      # self.last_state = state

      return action