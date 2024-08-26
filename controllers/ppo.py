from . import BaseController
import numpy as np
import torch
from PPO_control.ActorCritic import ActorCritic

STEER_RANGE = [-2, 2]
ACTION_DIM = 81

class Controller(BaseController):
  def __init__(self, policy_model: ActorCritic, device: torch.device) -> None:
    super().__init__()
    self.device = device
    self.policy_model = policy_model
    self.concat_state_history = []
    
    # For training
    self.observation_history = []
    self.target_history = []
    self.action_history = []
    self.action_prob_history = []
    self.value_history = []

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    steer = np.zeros_like(target_lataccel)

    if self.concat_state_history:
        self.concat_state_history[-1][:, :, -1] = current_lataccel[:, np.newaxis]
        past_obs = np.concatenate(self.concat_state_history[-self.policy_model.seq_len:], axis=1)
        target = np.column_stack([state, target_lataccel])
        past_obs = torch.from_numpy(past_obs).to(self.device)
        target = torch.from_numpy(target).to(self.device)        
        
        action, action_prob, value = self.policy_model.act(past_obs, target)
        steer = STEER_RANGE[0] + action * (STEER_RANGE[1] - STEER_RANGE[0]) / (ACTION_DIM - 1)

        self.observation_history.append(past_obs.cpu())
        self.target_history.append(target.cpu())
        self.action_history.append(action)
        self.action_prob_history.append(action_prob)
        self.value_history.append(value)

    self.concat_state_history.append(np.column_stack([state, steer, np.zeros_like(steer)])[:, np.newaxis])

    return action
