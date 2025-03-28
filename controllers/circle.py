from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    A PID controller modified to make the car follow a circular path.
    """
    def __init__(self, circle_radius=10.0, speed=5.0):
        # PID gains
        self.p = 0.1
        self.i = 0.05
        self.d = -0.03
        self.error_integral = 0
        self.prev_error = 0

        # Circular path parameters
        self.circle_radius = circle_radius  # meters
        self.speed = speed                  # m/s
        self.target_lataccel = (self.speed ** 2) / self.circle_radius  # Centripetal acceleration (v²/r)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Override target_lataccel with the circular path's required acceleration
        target_lataccel = self.target_lataccel  

        # PID control to track the circular path
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error

        steering_command = self.p * error + self.i * self.error_integral + self.d * error_diff
        return steering_command
