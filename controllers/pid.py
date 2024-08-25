from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.3
    self.i = 0.05
    self.d = -0.1
    self.error_integral = None
    self.prev_error = None

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      if not self.error_integral:
         self.error_integral = np.zeros_like(target_lataccel)
         self.prev_error = np.zeros_like(target_lataccel)
         
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      return self.p * error + self.i * self.error_integral + self.d * error_diff
