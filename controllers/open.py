from . import BaseController


class Controller(BaseController):
  """
  An open-loop controller
  """
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel
