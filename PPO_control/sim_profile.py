import sys
import os
from typing import Tuple, Callable, List
from pathlib import Path
import random
from functools import partial
sys.path.insert(1, os.path.abspath('..') )

import pandas as pd
import torch
import numpy as np

from PPO_Loss import *
from ActorCritic import ActorCritic
from ExperienceBuffer import PPOExperienceBuffer
from controllers import ppo
from tinyphysics_sim import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX

def find_shortest_path(files):
    lengths = [pd.read_csv(f).shape[0] for f in files]
    shortest_idx = np.argmin(lengths)
    return files[shortest_idx], lengths[shortest_idx]

def episode_rollout(model: TinyPhysicsModel, data_paths: List[str], buffer: PPOExperienceBuffer, policy_model: ActorCritic, n_episode: int, episode_len: int, path_boundary: float, device: torch.device):
    files = random.sample(data_paths, n_episode)
    controller = ppo.Controller(policy_model, device)

    # Prevent short paths
    while True:
        try:
            sim = TinyPhysicsSimulator(model, files, controller=controller, episode_len=episode_len + 1)
        except ValueError:
            shortest_file, shortest_length = find_shortest_path(files)
            print(f'Error: The requested array has an inhomogeneous shape after 1 dimensions. Shortest file: {shortest_file}, len: {shortest_length}')
            files = random.sample(data_paths, n_episode)
        else:
            break

    lat_cost, jerk_cost, total_cost = sim.rollout()

    # Calculate rewards
    deviation = np.abs(sim.target_lataccel_histories[:, CONTROL_START_IDX:] - sim.current_lataccel_histories[:, CONTROL_START_IDX:])
    rewards = 1 - deviation / path_boundary
    rewards[rewards < 0] = -1
    rewards = torch.from_numpy(rewards)
    if rewards.shape[1] != episode_len + 1:
        print(f"reward length doesn't match episode_len+1: rewards len:{len(rewards)}, episode_len+1:{episode_len+1}") 
        return

    buffer.reset()
    
    # Leave the value of last step for future reward estimation
    # Convert from list of episode_length of batch history to tensor of (batchsize, episode_len, ...)
    observation_history = torch.cat([x[:, None] for x in controller.observation_history[-episode_len-1: -1]], dim=1)
    target_history = torch.cat([x[:, None] for x in controller.target_history[-episode_len-1: -1]], dim=1)
    action_history = torch.column_stack(controller.action_history[-episode_len-1: -1])
    action_prob_history = torch.cat([x[:, None] for x in controller.action_prob_history[-episode_len-1: -1]], dim=1)
    value_history = torch.column_stack(controller.value_history[-episode_len-1:])

    buffer.batch_add_trajectory(observation_history,
                                target_history,
                                action_history,
                                action_prob_history,
                                rewards[:, :-1],
                                value_history)

    return buffer, rewards, lat_cost, jerk_cost, total_cost

if __name__ == '__main__':
    device = torch.device('cuda')
    tinyphysicsmodel = TinyPhysicsModel('../models/tinyphysics.onnx', debug=False)
    policy_model = ActorCritic(obs_dim=5, obs_seq_len=10, target_dim=4, action_dim=ppo.ACTION_DIM, has_continuous_action=False).to(device)
    data = Path('../data/')
    data_paths = sorted(data.iterdir())[5000:]
    data_paths = [str(x) for x in data_paths]
    experience_buffer = PPOExperienceBuffer(discount_factor=0.95, td_decay=0.85)

    # Run simulation
    n_episode = 384
    episode_len = 400
    path_boundary = 0.07
    sim_rollout = partial(episode_rollout, tinyphysicsmodel, data_paths, experience_buffer)

    buffer, rewards, lat_cost, jerk_cost, total_cost = sim_rollout(policy_model, n_episode, episode_len, path_boundary, device)
    print(f'average_reward: {rewards.mean(): .4f}, lat_cost: {lat_cost: .2f}, jerk_cost: {jerk_cost: .2f}')