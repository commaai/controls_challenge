import logging
from typing import Tuple, List, Union

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, target_dim: int, action_dim: int, has_continuous_action: bool):
        super().__init__()
        self.obs_dim = obs_dim
        self.target_dim = target_dim
        self.action_dim = action_dim
        self.has_continuous_action = has_continuous_action
        self.rng = np.random.default_rng()

        self.feature_lstm = nn.LSTM(input_size=self.obs_dim, hidden_size=32, batch_first=True)


        if self.has_continuous_action:
            self.actor = nn.Sequential(
                nn.Linear(32 + self.target_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(32 + self.target_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.action_dim),
                nn.Softmax(dim=-1)
            )

        self.critic = nn.Sequential(
            nn.Linear(32 + self.target_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Init all hidden layer weight to orthogonal weight with scale sqrt(2), bias to 2
        # Init value output layer weight with scale 1, policy output layer with scale 0.01
        for param in self.named_parameters():
            if param[0].endswith('weight'):
                nn.init.orthogonal_(param[1], gain=np.sqrt(2))
            elif param[0].endswith('bias'):
                nn.init.zeros_(param[1])
        nn.init.orthogonal_(list(self.critic.parameters())[-2], gain=1)
        nn.init.orthogonal_(list(self.actor.parameters())[-2], gain=0.01)
    
    def set_std(self, std: float):
        self.std = std
        if not self.has_continuous_action:
            logger.warning("Trying to set standard deviation for discrete action space.")

    def forward(self, past_obs, target):
        condition = self.feature_lstm(past_obs)[:, -1]
        x = torch.hstack([condition, target])
        action_logit = self.actor(x)
        value = self.critic(x.detach())

        return action_logit, value

    def act(self, past_obs: torch.Tensor, target: torch.Tensor) -> Tuple[List[Union[int, float]], torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.eval()
            action_logit, value = self.forward(past_obs, target)
            action_logit = action_logit.cpu().numpy()

        # Discrete case
        action = (action_logit.cumsum(axis=1) > self.rng.random(action_logit.shape[0])[:, np.newaxis]).argmax(axis=1) # Inverse transform sampling
        action_prob = action_logit[np.arange(len(action)), action]
        
        return action, action_prob, value.cpu().flatten().numpy()

