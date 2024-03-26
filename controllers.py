class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class ZeroController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return 0.


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3
