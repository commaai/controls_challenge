from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self, lag=0, alpha=1.6942529739501502, beta=0.0028692845935240664):
    self.alpha = alpha
    self.beta = beta
    self.lag = lag
    self.p = 0.3
    self.i = 0.03
    self.error_integral = 0

  def minimize(self, w_x, w_u, plan):
    d = 2 * self.alpha * w_x
    d[:-1] += 2 * w_u
    d[1:] += 2 * w_u

    A = np.diag(d) + np.diag(-2 * w_u, k=1) + np.diag(-2 * w_u, k=-1)
    b = 2 * self.alpha * (plan - self.beta) * w_x

    return np.linalg.solve(A, b)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if len(future_plan[0]) < self.lag:
      return 0

    steer_lataccel = target_lataccel - state[0]
    future_plan = np.array(future_plan)
    future_steer_lataccel = future_plan[0] - future_plan[1]
    future_steer_lataccel = np.concatenate(([steer_lataccel], future_steer_lataccel))
    error = (target_lataccel - current_lataccel)
    self.error_integral += error

    return 0.4 * (future_steer_lataccel[self.lag] - self.beta) / self.alpha + self.p * error + self.i * self.error_integral

    future_steer_lataccel = future_steer_lataccel[self.lag:]
    w_x = np.power(0.9, np.arange(len(future_steer_lataccel)))
    w_u = np.power(0.9, np.arange(len(future_steer_lataccel) - 1))
    
    return self.minimize(w_x, w_u, np.array(future_steer_lataccel))[0]