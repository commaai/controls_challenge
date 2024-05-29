class BaseController:
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    """
    Args:
      target_lataccel: The target lateral acceleration.
      current_lataccel: The current lateral acceleration.
      state: The current state of the vehicle.
      future_plan: The future plan for the next N frames.
    Returns:
      The control signal to be applied to the vehicle.
    """
    raise NotImplementedError
