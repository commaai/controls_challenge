from . import BaseController
import numpy as np
from scipy.signal import butter, lfilter

ACC_G = 9.81

class Controller(BaseController):
  """
  A PIDF controller with bycicle dynamic model feedforward
  """
  def __init__(self,):


    self.alpha = 0.9697
    self.lag = 2
    self.action_history = []
    self.error_history = []
    self.la_history = []
    self.last_state = None

    # Debug
    self.predict_la_history = []
    self.target_la_history = []
    self.delta_action_history = []
  
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
      def lowpass_filter(data):
        nyq = 0.5 * 10
        normal_cutoff = 2 / nyq
        b, a = butter(1, normal_cutoff, btype='low', analog=False)
        return lfilter(b, a, data)

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