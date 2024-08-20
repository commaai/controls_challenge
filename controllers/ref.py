from . import BaseController
import pandas as pd

CONTEXT_LENGTH = 20


class Controller(BaseController):
  """
  A controller that always outputs zero
  """
  def __init__(self, datafile) -> None:
    self.step = CONTEXT_LENGTH
    df = pd.read_csv(datafile)
    self.steerCommands = -df['steerCommand'].to_numpy()

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    command = self.steerCommands[self.step]
    self.step += 1
    return command
