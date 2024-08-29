import logging
from typing import Union, List, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np

from PPO_Loss import calculate_value_target_vec, generalized_advantage_estimation_vec

logger = logging.getLogger(__name__)

class PPOExperienceBuffer(Dataset):
    def __init__(self, discount_factor: float=0.99, td_decay: float=0.9) -> None:
        super().__init__()
        self.actions = []
        self.observations = []
        self.targets = []
        self.samples = torch.zeros((0, 4))
        self.discount_factor = discount_factor
        self.td_decay = td_decay
    
    def __len__(self) -> int:
        return self.samples.shape[0]
    
    def __getitem__(self, idx) -> Tuple[Union[int, float], torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actions[idx], self.observations[idx], self.targets[idx], self.samples[idx]

    def batch_add_trajectory(self, observation: torch.Tensor, target: torch.Tensor, action: torch.Tensor, action_p: torch.Tensor, reward: torch.Tensor, value_estimate: torch.Tensor) -> None:
        """
        Add a batch of trajectories to the buffer.

        Args:
            observation (batch_size, episode_len): Batch observation at each time step
            target (batch_size, episode_len): Batch target at each time step
            action (batch_size, episode_len): Batch actions taken during the episode.
            action_p (batch_size, episode_len): Batch action probabilities corresponding to each action.
            reward (batch_size, episode_len): Batch rewards received for each step.
            value_estimate (batch_size, episode_len+1): Batch value estimates, including the estimate for the final state.

        Raises:
            ValueError: If input lengths are inconsistent with the episode length.

        Note:
            The length of value_estimate should be one more than the other inputs to include the final state estimate.
        """
        
        episode_len = action.shape[1]

        if action_p.shape[1] != episode_len:
            raise ValueError(f'action_p length mismatch. episode length: {episode_len}, action_p length: {action_p.shape[1]}')
        if reward.shape[1] != episode_len:
            raise ValueError(f'reward length mismatch. episode length: {episode_len}, reward length: {reward.shape[1]}')
        if value_estimate.shape[1] != episode_len + 1:
            raise ValueError(f'value_estimate length mismatch. expected length: {episode_len + 1}, actual length: {len(value_estimate)}')
        
        self.actions += action.flatten().tolist()
        self.observations += [x for x in observation.flatten(0, 1)]
        self.targets += [x for x in target.flatten(0, 1)]

        value_target = calculate_value_target_vec(reward, value_estimate, self.discount_factor)
        gae = generalized_advantage_estimation_vec(reward, value_estimate, self.discount_factor, self.td_decay)
        
        batch_trajectory = torch.cat([action_p.unsqueeze(2), value_estimate[:, :-1].unsqueeze(2), value_target.unsqueeze(2), gae.unsqueeze(2)], dim=2)
        self.samples = torch.vstack([self.samples, batch_trajectory.flatten(0, 1)])

    def reset(self):
        self.actions = []
        self.observations = []
        self.targets = []
        self.samples = torch.zeros((0, 4))