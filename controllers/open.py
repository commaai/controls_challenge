from . import BaseController


class Controller(BaseController):
  """
  An open-loop controller
  """
  def update(self, target_lataccel, current_lataccel, state, target_future):
    return target_lataccel
