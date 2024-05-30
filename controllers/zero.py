from . import BaseController


class Controller(BaseController):
  """
  A controller that always outputs zero
  """
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    return 0.0
