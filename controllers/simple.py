from . import BaseController


class Controller(BaseController):
  """
  A simple controller that is the error between the target and current lateral acceleration times some factor
  """
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    return (target_lataccel - current_lataccel) * 0.3
