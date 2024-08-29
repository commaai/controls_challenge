import sys
import os
from typing import Tuple, Callable, List
from pathlib import Path
import argparse
import json
import random
from functools import partial

from torch._prims_common import check
sys.path.insert(1, os.path.abspath('..') )

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from PPO_Loss import *
from ActorCritic import ActorCritic
from ExperienceBuffer import PPOExperienceBuffer
from controllers import ppo
from tinyphysics_sim import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX, MAX_ACC_DELTA

def find_shortest_path(dfs):
    return np.min([df.shape[0] for df in dfs])

def episode_rollout(model: TinyPhysicsModel, dfs: List[pd.DataFrame], buffer: PPOExperienceBuffer, policy_model: ActorCritic, n_episode: int, episode_len: int, path_boundary: float, device: torch.device):
    sample_dfs = random.sample(dfs, n_episode)
    controller = ppo.Controller(policy_model, device)

    # Prevent short paths
    while True:
        try:
            sim = TinyPhysicsSimulator(model, sample_dfs, controller=controller, episode_len=episode_len + 1)
        except ValueError:
            shortest_length = find_shortest_path(sample_dfs)
            print(f'Error: The requested array has an inhomogeneous shape after 1 dimensions. Shortest len: {shortest_length}')
            sample_dfs = random.sample(dfs, n_episode)

        else:
            break

    lat_cost, jerk_cost, total_cost = sim.rollout()

    # Calculate rewards
    deviation = np.abs(sim.target_lataccel_histories[:, CONTROL_START_IDX + policy_model.obs_seq_len:] - sim.current_lataccel_histories[:, CONTROL_START_IDX + policy_model.obs_seq_len:])
    jerk = np.abs(np.diff(sim.current_lataccel_histories[:, CONTROL_START_IDX + policy_model.obs_seq_len - 1:], axis=1))
    d_rewards = 1 - deviation / path_boundary
    d_rewards[d_rewards < 0] = 0
    rewards = torch.from_numpy(d_rewards)

    if rewards.shape[1] != episode_len + 1 - policy_model.obs_seq_len:
        print(f"reward length doesn't match episode_len+1: rewards len:{len(rewards)}, episode_len+1:{episode_len+1}") 
        return

    buffer.reset()
    
    # Leave the value of last step for future reward estimation
    # Convert from list of episode_length of batch history to tensor of (batchsize, episode_len, ...)
    observation_history = torch.cat([x[:, None] for x in controller.observation_history[-episode_len - 1 +policy_model.obs_seq_len: -1]], dim=1)
    target_history = torch.cat([x[:, None] for x in controller.target_history[-episode_len-1+policy_model.obs_seq_len: -1]], dim=1)
    action_history = torch.column_stack(controller.action_history[-episode_len-1+policy_model.obs_seq_len: -1])
    action_prob_history = torch.cat([x[:, None] for x in controller.action_prob_history[-episode_len-1+policy_model.obs_seq_len: -1]], dim=1)
    value_history = torch.column_stack(controller.value_history[-episode_len-1+policy_model.obs_seq_len:])

    buffer.batch_add_trajectory(observation_history,
                                target_history,
                                action_history,
                                action_prob_history,
                                rewards[:, :-1],
                                value_history)

    return buffer, rewards, lat_cost, jerk_cost, total_cost

def train(policy_model: ActorCritic, optimizer: optim.Optimizer, run_rollout: Callable, episode_len: int, path_boundary: float, n_steps: int, n_episode: int, n_epoch: int, batch_size: int, clip_eps: float, c1: float, c2: float, device: torch.device, checkpoint=None):
    reward_history = []
    lat_cost_history = []
    jerk_cost_history = []

    total_loss_history = []
    actor_obj_history = []
    critic_loss_history = []
    entropy_obj_history = []

    start_step = 0
    
    if checkpoint:
        start_step = checkpoint['step']
        policy_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        reward_history = checkpoint['reward_history']
        lat_cost_history = checkpoint['lat_cost_history']
        jerk_cost_history = checkpoint['jerk_cost_history']
        total_loss_history = checkpoint['total_loss_history']
        actor_obj_history = checkpoint['actor_obj_history']
        critic_loss_history = checkpoint['critic_loss_history']
        entropy_obj_history = checkpoint['entropy_obj_history']

    for i_step in range(start_step, n_steps):
        # Run simulation
        buffer, rewards, lat_cost, jerk_cost, total_cost = run_rollout(policy_model, n_episode, episode_len, path_boundary, device)
        print(f'Training step {i_step:>7}: average_reward: {rewards.mean(): .4f}, lat_cost: {lat_cost: .2f}, jerk_cost: {jerk_cost: .2f}')
        # Record policy's performance at each step
        reward_history.append(rewards.mean().item())
        lat_cost_history.append(lat_cost)
        jerk_cost_history.append(jerk_cost)

        # Start training loops
        data_loader = DataLoader(buffer, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        policy_model.train()
        for i_epoch in range(n_epoch):
            print(f'Training step {i_step:>7}, epoch {i_epoch:>3}:', end='')

            epoch_total_loss_history = []
            epoch_actor_obj_history = []
            epoch_critic_loss_history = []
            epoch_entropy_obj_history = []

            for data in data_loader:
                actions, observations, targets, transitions = data
                actions = actions.type(torch.int)
                observations = observations.to(device)
                targets = targets.to(device)
                transitions = transitions.to(device)

                optimizer.zero_grad()

                action_prob_new, value_estimates = policy_model(observations, targets)
                action_prob_new = action_prob_new[torch.arange(actions.shape[0]), actions]
                value_estimates = value_estimates.flatten()

                loss, actor_obj, critic_loss, entropy_obj = ppo_loss(action_prob_new, transitions[:, 0], transitions[:, 3], value_estimates, transitions[:, 2], clip_eps, c1, c2)
                loss.backward()
                optimizer.step()

                epoch_total_loss_history.append(loss.item())
                epoch_actor_obj_history.append(actor_obj)
                epoch_critic_loss_history.append(critic_loss)
                epoch_entropy_obj_history.append(entropy_obj)

            total_loss_history.append(np.mean(epoch_total_loss_history))
            actor_obj_history.append(np.mean(epoch_actor_obj_history))
            critic_loss_history.append(np.mean(epoch_critic_loss_history))
            entropy_obj_history.append(np.mean(epoch_entropy_obj_history))

            print(f' total_loss: {total_loss_history[-1]: 0.3f}, actor_obj: {actor_obj_history[-1]: 0.3f}, critic_loss: {critic_loss_history[-1]: 0.3f}, entropy_obj: {entropy_obj_history[-1]: 0.3f}')
        
        if i_step % 10 == 9:
            torch.save({
                'step': i_step,
                'model_state_dict': policy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward_history': reward_history,
                'lat_cost_history': lat_cost_history,
                'jerk_cost_history': jerk_cost_history,
                'total_loss_history': total_loss_history,
                'actor_obj_history': actor_obj_history,
                'critic_loss_history': critic_loss_history,
                'entropy_obj_history': entropy_obj_history
            }, f'./checkpoints/exp{model_id}_step{i_step + 1}.pt')

    return reward_history, lat_cost_history, jerk_cost_history, total_loss_history, actor_obj_history, critic_loss_history, entropy_obj_history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=False)
    args = parser.parse_args()
    
    with open(args.config_path) as f:
        config = json.load(f)

    checkpoint = None
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)

    print(f"Start training with checkpoint: {args.checkpoint_path}, config: {args.config_path}")

    global model_id
    model_id = config['model_id']
    n_episode = config['n_episode']
    episode_len = config['episode_len']
    path_boundary = config['path_boundary']
    lr = config['lr']
    batch_size = config['batch_size']
    n_epoch = config['n_epoch']
    n_step = config['n_step']
    c1 = config['c1']
    c2 = config['c2']
    clip_eps = config['clip_eps']
    discount_factor = config['discount_factor']
    td_decay = config['td_decay']

    device = torch.device('cuda')
    tinyphysicsmodel = TinyPhysicsModel('../models/tinyphysics.onnx', debug=False)
    policy_model = ActorCritic(obs_dim=4, obs_seq_len=10, target_dim=31, action_dim=ppo.ACTION_DIM, has_continuous_action=False).to(device)
    data = Path('../data/')
    data_paths = sorted(data.iterdir())[5000:]
    data_paths = [str(x) for x in data_paths]
    dfs = [pd.read_csv(f) for f in data_paths]
    experience_buffer = PPOExperienceBuffer(discount_factor, td_decay)

    optimizer = optim.Adam(policy_model.parameters(), lr, eps=1e-5)

    sim_rollout = partial(episode_rollout, tinyphysicsmodel, dfs, experience_buffer)

    result = train(policy_model, optimizer, sim_rollout, episode_len, path_boundary, n_step, n_episode, n_epoch, batch_size, clip_eps, c1, c2, device, checkpoint)
    np.save(f'results/result_{model_id}.npy', np.array(result, dtype=object), allow_pickle=True)
    