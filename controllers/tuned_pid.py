from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simplified tuned PID controller with feed-forward term
  
  Based on the baseline PID controller but adds a feed-forward term
  Feed-forward term: k_ff * target_lataccel / v_ego**2
  """
  def __init__(self):
    # Optimized PID gains from Optuna (trial 25)
    self.p = 0.21942091482230813  # Proportional gain
    self.i = 0.09438349898803905  # Integral gain
    self.d = -0.06219594603865014  # Derivative gain
    
    # Add feed-forward gain
    self.k_ff = 0.02737288447587737
    
    # State variables
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      # Extract vehicle velocity for feed-forward term
      v_ego = state.v_ego
      
      # Calculate PID terms (same as baseline)
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      
      # Calculate feed-forward term (avoid division by zero)
      feed_forward = 0
      if v_ego > 0.1:  # Only apply feed-forward when moving
          feed_forward = self.k_ff * target_lataccel / (v_ego**2)
      
      # Combine PID and feed-forward
      return self.p * error + self.i * self.error_integral + self.d * error_diff + feed_forward 