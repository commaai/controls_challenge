from . import BaseController
import numpy as np
import torch
from PPO_control.ActorCritic import ActorCritic

STEER_RANGE = [-1, 1]
ACTION_DIM = 81

class Controller(BaseController):
  def __init__(self, policy_model: ActorCritic, device: torch.device, eval: bool=False) -> None:
    super().__init__()
    self.device = device
    self.eval = eval
    self.policy_model = policy_model
    self.concat_state_history = []

    # self.p = 0.3
    # self.i = 0.05
    # self.d = -0.1
    # self.error_integral = None
    # self.prev_error = None

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
        # if len(self.concat_state_history) < self.policy_model.obs_seq_len:
        #   if self.error_integral is None:
        #     self.error_integral = np.zeros_like(target_lataccel)
        #     self.prev_error = np.zeros_like(target_lataccel)
            
        #   error = (target_lataccel - current_lataccel)
        #   self.error_integral += error
        #   error_diff = error - self.prev_error
        #   self.prev_error = error
        #   steer = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # else:
        past_obs = np.concatenate(self.concat_state_history[-self.policy_model.obs_seq_len:], axis=1)
        past_obs = torch.from_numpy(past_obs).float().to(self.device)

        future_plan = future_plan[:, :9, [0, 1, 3]]
        target = np.zeros((state.shape[0], 31))
        target[:, 0] = current_lataccel
        target[:, 1:3] = state[:, :2]
        target[:, 3] = target_lataccel
        target[:, 4: future_plan.shape[1] * future_plan.shape[2] + 4] = future_plan.reshape((future_plan.shape[0], -1))
        target = torch.from_numpy(target).float().to(self.device)

        action, action_prob, value = self.policy_model.act(past_obs, target, self.eval)
        steer = action

        self.observation_history.append(past_obs.cpu())
        self.target_history.append(target.cpu())
        self.action_history.append(torch.from_numpy(action))
        self.action_prob_history.append(torch.from_numpy(action_prob))
        self.value_history.append(torch.from_numpy(value))

    self.concat_state_history.append(np.column_stack([state[:, :2], steer, np.zeros_like(steer)])[:, np.newaxis])

    return steer
