from . import BaseController
import numpy as np
import math

class Controller(BaseController):
  """
  Proportional-Integral-Derivative (PID) controller
  """
  def __init__(self,):
    self.counter = 0
    self.p = 0.3
    self.i = 0.07
    self.d = -0.1
    
    self.error_integral = 0
    self.max_integral = 5
    self.prev_error = 0
    
    
    self.steer_factor = 13 # lat accel to steer command factor
    self.steer_sat_v = 20 # saturate v measurements for steerign
        

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      ## state hass roll_lataccel, v_eg, a_ego
      
      # reset error history before start:
      self.counter += 1
      if self.counter == 81:
        self.error_integral = 0
        self.prev_error = 0
      
      # because  the response time is slow:
      if len(future_plan) >= 5:
            target_lataccel = np.average([target_lataccel] + future_plan[0:3], weights = [2, 2, 2, 2])     
      
      # normal pid
      error = target_lataccel - current_lataccel
      self.error_integral += error
      error_difference = error - self.prev_error
      self.prev_error = error 
      
      self.error_integral = max(min(self.error_integral, self.max_integral), -self.max_integral)
                           

      # sclae down p for high speeds and d for aceleration
      p = self.p / (1 + 0.05 * abs(state.v_ego))
      d = self.d * (1 - 0.1 * abs(state.a_ego))
      
      # pid input   
      u_pid = p * error + self.i * self.error_integral + d * error_difference
      
      
      # estimate steer command with target target lateral acceleration  
      steer_lataccel_target = (target_lataccel - state.roll_lataccel)
      steer_command = (steer_lataccel_target * self.steer_factor) / max(self.steer_sat_v, state.v_ego)
      
      # tanh to dampen extremes.
      steer_command = 2 * np.tanh(steer_command/2)
      
      return (u_pid + 0.7*steer_command)