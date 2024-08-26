from typing import Tuple

import torch

def ppo_loss(action_p: torch.Tensor, action_p_old: torch.Tensor, advantage: torch.Tensor, value: torch.Tensor, value_target: torch.Tensor, clip_eps: float, c1: float, c2: float) -> Tuple[torch.Tensor, float, float, float]:
    eps = 1e-8 if action_p.dtype is torch.float else 1e-6
    r = action_p / (action_p_old + eps)
    L_clip = torch.mean(torch.min(r * advantage, torch.clip(r, 1 - clip_eps, 1 + clip_eps) * advantage))
    L_value = torch.mean((value - value_target) ** 2)
    L_entropy = torch.mean(-torch.log(action_p))

    return -L_clip + c1 * L_value - c2 * L_entropy, L_clip.item(), L_value.item(), L_entropy.item() 

def generalized_advantage_estimation(reward: torch.Tensor, value: torch.Tensor, discount_factor: float, decay: float) -> torch.Tensor:
    '''
    reward: batched rewards for each time step (batch_size, T)
    value: batched value estimation for each time step plus last state (batch_size, T+1)
    '''
    eps = 1e-8 if reward.dtype is torch.float else 1e-6

    advantage = torch.zeros_like(reward)
    last_adv = 0
    for t in reversed(range(reward.shape[1])):
        delta = reward[:, t] + discount_factor * value[:, t + 1] - value[:, t]
        advantage[:, t] = delta + discount_factor * decay * last_adv
        last_adv = advantage[:, t]
    
    # Normalize across batch
    advantage = (advantage - advantage.mean()) / (advantage.std() + eps)

    return advantage

def calculate_value_target(reward: torch.Tensor, value: torch.Tensor, discount_factor: float) -> torch.Tensor:
    '''
    reward: batched rewards for each time step (batch_size, T)
    value: batched value estimation for each time step plus last state (batch_size, T+1)
    '''
    eps = 1e-8 if reward.dtype is torch.float else 1e-6

    v_target = torch.zeros_like(reward)
    last_reward = value[:, -1]
    for t in reversed(range(reward.shape[1])):
        v_target[:, t] = reward[:, t] + discount_factor * last_reward
        last_reward = v_target[:, t]

    v_target = (v_target - v_target.mean()) / (v_target.std() + eps)

    return v_target

def calculate_value_target_vec(reward: torch.Tensor, value: torch.Tensor, discount_factor: float) -> torch.Tensor:
    '''
    reward: batched rewards for each time step (batch_size, T)
    value: batched value estimation for each time step plus last state (batch_size, T+1)
    '''
    T = reward.shape[1]
    eps = 1e-8 if reward.dtype is torch.float else 1e-6

    # Calculate discounted sum of rewards
    discount_factors = discount_factor ** torch.arange(T, dtype=torch.float32, device=reward.device)
    future_discounted_rewards = torch.cumsum((reward * discount_factors.unsqueeze(0)).flip(1), dim=1).flip(1)
    v_target = future_discounted_rewards / discount_factors

    # Add the discounted final value estimate
    v_target = v_target + torch.outer(value[:, -1], (discount_factor * discount_factors).flip(0))

    v_target = (v_target - v_target.mean()) / (v_target.std() + eps)
    
    return v_target

def generalized_advantage_estimation_vec(reward: torch.Tensor, value: torch.Tensor, discount_factor: float, decay: float) -> torch.Tensor:
    '''
    reward: batched rewards for each time step (batch_size, T)
    value: batched value estimation for each time step plus last state (batch_size, T+1)
    '''
    T = reward.shape[1]
    eps = 1e-8 if reward.dtype is torch.float else 1e-6
    
    # Calculate delta
    delta = reward + discount_factor * value[:, 1:] - value[:, :-1]
    
    # Calculate discount factors
    discount_factors = (discount_factor * decay) ** torch.arange(T, dtype=torch.float32, device=reward.device)
    
    # Calculate advantages
    advantage = torch.cumsum((delta * discount_factors.unsqueeze(0)).flip(1), dim=1).flip(1)
    advantage = advantage / discount_factors.unsqueeze(0)

    # Normalize across batch
    advantage = (advantage - advantage.mean()) / (advantage.std() + eps)
    
    return advantage