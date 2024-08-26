import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np 
from PPO_Loss import ppo_loss
from ActorCritic import ActorCritic

from typing import Callable

def train_combined_actorcritic(policy_model: ActorCritic, optimizer: optim.Optimizer, run_rollout: Callable, n_steps: int, n_episode: int, n_epoch: int, batch_size: int, clip_eps: float, c1: float, c2: float, device: torch.device):
    score_history = []
    total_loss_history = []
    actor_obj_history = []
    critic_loss_history = []
    entropy_obj_history = []

    for i_step in range(n_steps):
        buffer, average_score = run_rollout(policy_model, n_episode, device)
        
        # Record policy's performance at each step
        score_history.append(average_score)
        
        data_loader = DataLoader(buffer, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        for i_epoch in range(n_epoch):
            print(f'Training step {i_step:>3}, epoch {i_epoch:>3}: average_score: {average_score: .1f}', end='')

            epoch_total_loss_history = []
            epoch_actor_obj_history = []
            epoch_critic_loss_history = []
            epoch_entropy_obj_history = []
            
            for i_batch, data in enumerate(data_loader):
                actions, observations, transitions = data
                actions = actions.type(torch.int)
                observations = observations.to(device)
                
                optimizer.zero_grad()
                
                action_prob_new, value_estimates = policy_model(observations)
                action_prob_new = action_prob_new.cpu()[torch.arange(actions.shape[0]), actions]
                value_estimates = value_estimates.cpu().flatten()

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
    
    return score_history, total_loss_history, actor_obj_history, critic_loss_history, entropy_obj_history
