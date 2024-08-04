from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    A simple PID controller
    """
    def __init__(self,):
        self.p = 0.4  # Increased proportional gain
        self.i = 0.1  # Increased integral gain
        self.d = -0.05  # Reduced derivative gain
        self.error_integral = 0
        self.prev_error = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff

