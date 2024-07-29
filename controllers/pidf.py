from . import BaseController
import numpy as np

ACC_G = 9.81

class Controller(BaseController):
  """
  A PIDF controller with bycicle dynamic model feedforward
  """
  def __init__(self, p, i, d, f):
    self.p = p
    self.i = i
    self.d = d
    self.f = f
    self.error_integral = 0
    self.prev_error = 0

    self.steer_command_history = []
    self.current_la_history = []
    self.state_history = []
    self.beta0 = -0.0769574481921961
    self.beta1 = 0.0027094010571292247

  def update_fit(self):
    '''
    Fit parameters beta0, beta1 of a front steering bycicle model with simple linear regression:
    $a_{lat} = \beta_0 + \beta_1 * v^2 * sin(\gamma)$
    where gamma is the steer command.

    beta1 accounts for steer ratio and vehicle length,
    as this model assume a linear relationship between steer command and steering angle,
    and that we can approximate sin(x) with x with small angles.

    At the same time, L is just another multiplicative coefficient
    so we can estimate it and steer ratio together.
    
    beta0 would account for other factors like vehicle slip,
    which we assume to be a constant with small steer angle.

    The parameters of this model can be easily estimated using simple linear regression.
    '''

    commands = np.array(self.steer_command_history)
    roll_la = np.array([x[0] for x in self.state_history])
    la = np.array(self.current_la_history) - roll_la
    vEgo = np.array([x[1] for x in self.state_history])

    pred_la = vEgo ** 2 * np.sin(commands)

    print(f'commands: {commands}')
    
    self.beta1 = np.sum((pred_la - pred_la.mean()) * (la - la.mean())) / np.sum((pred_la - pred_la.mean()) ** 2)
    self.beta0 = la.mean() - self.beta1 * pred_la.mean()

    print(f'beta0: {self.beta0}, beta1: {self.beta1}')
    
  def predict_steer(self, target_steer_la, vEgo):
    return np.arcsin((target_steer_la - self.beta0) / self.beta1 / vEgo ** 2)
    
  def update(self, target_lataccel, current_lataccel, state, future_plan):
      # Update state history
      if self.steer_command_history:
        self.current_la_history.append(current_lataccel)
      # if len(self.current_la_history) > 2:
      #   self.update_fit()
      self.state_history.append(state)

      # Feedforward
      target_steer_la = target_lataccel - state[0]
      pred_steer = self.predict_steer(target_steer_la, state[1])

      # PID Feedback
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error

      command = self.f * pred_steer + self.p * error + self.i * self.error_integral + self.d * error_diff
      self.steer_command_history.append(command)

      return command