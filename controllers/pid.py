from . import BaseController
import numpy as np

class Controller(BaseController):
    """
    A simple PID controller that makes the car dance.
    """
    def __init__(self):
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0
        self.dance_step = 0
        self.dance_moves = [
            self.dance_move_1,
            self.dance_move_2,
            self.dance_move_3,
            self.dance_move_4,
            self.dance_move_5
        ]
        self.current_move = 0
        self.move_duration = 100  # Duration of each dance move in update cycles
        self.move_counter = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        if self.move_counter >= self.move_duration:
            self.current_move = (self.current_move + 1) % len(self.dance_moves)
            self.move_counter = 0

        # Get the current dance move function and execute it
        dance_move = self.dance_moves[self.current_move]
        steering_command = dance_move(target_lataccel, current_lataccel, state, future_plan)

        self.move_counter += 1
        return steering_command

    def dance_move_1(self, target_lataccel, current_lataccel, state, future_plan):
        # Dance Move 1: Sway left and right
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff + 0.5 * np.sin(self.move_counter * 0.1)

    def dance_move_2(self, target_lataccel, current_lataccel, state, future_plan):
        # Dance Move 2: Sharp left turns
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff + 0.8 * (1 if self.move_counter % 20 < 10 else -1)

    def dance_move_3(self, target_lataccel, current_lataccel, state, future_plan):
        # Dance Move 3: Smooth oscillations
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff + 0.3 * np.sin(self.move_counter * 0.2)

    def dance_move_4(self, target_lataccel, current_lataccel, state, future_plan):
        # Dance Move 4: Random sharp turns
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff + 0.6 * np.random.uniform(-1, 1)

    def dance_move_5(self, target_lataccel, current_lataccel, state, future_plan):
        # Dance Move 5: Gradual left-right sway
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff + 0.4 * np.sin(self.move_counter * 0.05)
