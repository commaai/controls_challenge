import logging
from typing import Tuple, List, Union

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, obs_seq_len: int, target_dim: int, action_dim: int, has_continuous_action: bool, action_scale: float=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_seq_len = obs_seq_len
        self.target_dim = target_dim
        self.action_dim = action_dim
        self.has_continuous_action = has_continuous_action
        self.action_scale = action_scale
        self.rng = np.random.default_rng()

        self.feature_lstm = nn.LSTM(input_size=self.obs_dim, hidden_size=16, num_layers=2, batch_first=True)

        if self.has_continuous_action:
            self.actor = nn.Sequential(
                nn.Linear(16 + self.target_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, self.action_dim),
                nn.Tanh()
            )
            self.log_std = nn.Parameter(torch.zeros((self.action_dim), dtype=torch.float32))
        else:
            self.actor = nn.Sequential(
                nn.Linear(16 + self.target_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, self.action_dim),
                nn.Softmax(dim=-1)
            )

        self.critic = nn.Sequential(
            nn.Linear(16 + self.target_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Init all hidden layer weight to orthogonal weight with scale sqrt(2), bias to 2
        # Init value output layer weight with scale 1, policy output layer with scale 0.01
        for param in self.named_parameters():
            if param[0].endswith('weight'):
                if param[0].startswith('feature_lstm'):
                    nn.init.orthogonal_(param[1], gain=1)
                else:
                    nn.init.orthogonal_(param[1], gain=np.sqrt(2))
            elif param[0].endswith('bias'):
                nn.init.zeros_(param[1])
        nn.init.orthogonal_(list(self.critic.parameters())[-2], gain=1)
        nn.init.orthogonal_(list(self.actor.parameters())[-2], gain=0.01)

    def forward(self, past_obs, target):
        condition, _ = self.feature_lstm(past_obs)
        condition = condition[:, -1]
        x = torch.hstack([condition, target])
        action_logit = self.actor(x)
        value = self.critic(x)

        if self.has_continuous_action:
            return Normal(action_logit.flatten(), torch.exp(self.log_std)), value
        else:
            return action_logit, value

    def act(self, past_obs: torch.Tensor, target: torch.Tensor, eval: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            self.eval()
            action_logit, value = self.forward(past_obs, target)
        
            if self.has_continuous_action:
                # Continuous case
                if eval:
                    action = action_logit.mean
                    action_prob = torch.exp(action_logit.log_prob(action)).cpu().numpy()
                    action = action.cpu().numpy() * self.action_scale
                else:
                    action = action_logit.sample()
                    action_prob = torch.exp(action_logit.log_prob(action)).cpu().numpy()
                    action = action.cpu().numpy() * self.action_scale
            else:
                # Discrete case
                action_logit = action_logit.cpu().numpy()
                if eval:
                    action = action_logit.argmax(axis=1)
                    action_prob = action_logit[np.arange(len(action)), action]
                else:
                    action = (action_logit.cumsum(axis=1) > self.rng.random(action_logit.shape[0])[:, np.newaxis]).argmax(axis=1) # Inverse transform sampling
                    action_prob = action_logit[np.arange(len(action)), action]
            
        return action, action_prob, value.flatten().cpu().numpy()

