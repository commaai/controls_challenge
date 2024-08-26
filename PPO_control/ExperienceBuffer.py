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


    def add_trajectory(self, observation: List[torch.Tensor], action: List[Union[int, float]], action_p: List[float], reward: List[Union[int, float]], value_estimate: List[float]) -> None:
        """
        Add a trajectory to the buffer.

        Args:
            action (episode_len): List of actions taken during the episode.
            action_p (episode_len): List of action probabilities corresponding to each action.
            reward (episode_len): List of rewards received for each step.
            value_estimate (episode_len+1): List of value estimates, including the estimate for the final state.

        Raises:
            ValueError: If input lengths are inconsistent with the episode length.

        Note:
            The length of value_estimate should be one more than the other inputs to include the final state estimate.
        """

        episode_len = len(action)

        if len(action_p) != episode_len:
            raise ValueError(f'action_p length mismatch. episode length: {episode_len}, action_p length: {len(action_p) }')
        if len(reward) != episode_len:
            raise ValueError(f'reward length mismatch. episode length: {episode_len}, reward length: {len(reward)}')
        if len(value_estimate) != episode_len + 1:
            raise ValueError(f'value_estimate length mismatch. expected length: {episode_len + 1}, actual length: {len(value_estimate)}')
        
        self.actions += action
        self.observations += observation
        action_p = torch.tensor(action_p)
        reward = torch.tensor(reward)
        value_estimate = torch.tensor(value_estimate)

        value_target = calculate_value_target_vec(reward.unsqueeze(0), value_estimate.unsqueeze(0), self.discount_factor).squeeze()
        gae = generalized_advantage_estimation_vec(reward.unsqueeze(0), value_estimate.unsqueeze(0), self.discount_factor, self.td_decay).squeeze()
        
        trajectory = torch.column_stack([action_p, value_estimate[:-1], value_target, gae])
        self.samples = torch.vstack([self.samples, trajectory])


    def batch_add_trajectory(self, observation: List[torch.Tensor], target: List[torch.Tensor], action: List[torch.Tensor], action_p: List[torch.Tensor], reward: List[torch.Tensor], value_estimate: List[torch.Tensor]) -> None:
        """
        Add a batch of trajectories to the buffer.

        Args:
            observation (episode_len, batch_size): List of batch observation at each time step
            target (episode_len, batch_size): List of batch target at each time step
            action (episode_len, batch_size): List of batch actions taken during the episode.
            action_p (episode_len, batch_size): List of batch action probabilities corresponding to each action.
            reward (episode_len, batch_size): List of batch rewards received for each step.
            value_estimate (episode_len+1, batch_size): List of batch value estimates, including the estimate for the final state.

        Raises:
            ValueError: If input lengths are inconsistent with the episode length.

        Note:
            The length of value_estimate should be one more than the other inputs to include the final state estimate.
        """
        
        episode_len = len(action)

        if len(action_p) != episode_len:
            raise ValueError(f'action_p length mismatch. episode length: {episode_len}, action_p length: {len(action_p) }')
        if len(reward) != episode_len:
            raise ValueError(f'reward length mismatch. episode length: {episode_len}, reward length: {len(reward)}')
        if len(value_estimate) != episode_len + 1:
            raise ValueError(f'value_estimate length mismatch. expected length: {episode_len + 1}, actual length: {len(value_estimate)}')
        
        self.actions += [action[t][i] for i in range(len(action[0])) for t in range(len(action))]
        self.observations += [observation[t][i] for i in range(len(observation[0])) for t in range(len(observation))]
        self.targets += []  #TODO
        action_p = torch.tensor(action_p).T
        reward = torch.tensor(reward).T
        value_estimate = torch.tensor(value_estimate).T

        value_target = calculate_value_target_vec(reward, value_estimate, self.discount_factor)
        gae = generalized_advantage_estimation_vec(reward, value_estimate, self.discount_factor, self.td_decay)
        
        batch_trajectory = torch.cat([action_p.unsqueeze(2), value_estimate[:, :-1].unsqueeze(2), value_target.unsqueeze(2), gae.unsqueeze(2)], dim=2)
        self.samples = torch.vstack([self.samples, batch_trajectory.flatten(0, 1)])

    def reset(self):
        self.actions = []
        self.samples = torch.tensor([[]])