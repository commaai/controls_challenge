from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import BaseController


torch.set_num_threads(1)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "bc_student.pt"
_MODEL_CACHE: dict[Path, BCModel] = {}


class BCModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.past_encoder = nn.Sequential(
      nn.Linear(40, 256),
      nn.GELU(),
      nn.Linear(256, 128),
      nn.GELU(),
      nn.Linear(128, 128),
      nn.GELU(),
    )
    self.future_encoder = nn.Sequential(
      nn.Linear(200, 1024),
      nn.GELU(),
      nn.Linear(1024, 512),
      nn.GELU(),
      nn.Linear(512, 256),
      nn.GELU(),
    )
    self.current_encoder = nn.Sequential(
      nn.Linear(5, 64),
      nn.GELU(),
      nn.Linear(64, 64),
      nn.GELU(),
    )
    self.head = nn.Sequential(
      nn.Linear(450, 512),
      nn.GELU(),
      nn.Linear(512, 512),
      nn.GELU(),
      nn.Linear(512, 256),
      nn.GELU(),
      nn.Linear(256, 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    current = x[:, 0:5]
    past = x[:, 5:45]
    future = x[:, 45:245]
    timestep = x[:, 245:247]
    combined = torch.cat([
      self.past_encoder(past),
      self.future_encoder(future),
      self.current_encoder(current),
      timestep,
    ], dim=1)
    return torch.tanh(self.head(combined)).squeeze(-1) * 2.0


class Controller(BaseController):
  CONTEXT_LENGTH = 20
  FUTURE_LENGTH = 50
  INPUT_SIZE = 247
  CONTROL_START_IDX = 100
  SEGMENT_LENGTH = 400

  def __init__(self, model_path: str | None = None) -> None:
    self.model_path = Path(model_path) if model_path is not None else MODEL_PATH
    self.model: BCModel | None = None
    self.device = torch.device("cpu")
    self.past_actions = np.zeros(self.CONTEXT_LENGTH, dtype=np.float32)
    self.past_lataccels = np.zeros(self.CONTEXT_LENGTH, dtype=np.float32)
    self.total_steps = 0
    self.control_steps = 0

  def _load_model(self) -> None:
    resolved = self.model_path.resolve()
    model = _MODEL_CACHE.get(resolved)
    if model is None:
      model = BCModel().to(self.device)
      model.load_state_dict(torch.load(resolved, map_location=self.device, weights_only=True))
      model.eval()
      _MODEL_CACHE[resolved] = model
    self.model = model

  @property
  def _in_control(self) -> bool:
    return self.total_steps >= (self.CONTROL_START_IDX - self.CONTEXT_LENGTH)

  def _pad_future(self, values: list[float], current_value: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size >= self.FUTURE_LENGTH:
      return arr[:self.FUTURE_LENGTH]
    if arr.size > 0:
      pad_value = arr[-1]
    else:
      pad_value = np.float32(current_value)
    padding = np.full(self.FUTURE_LENGTH - arr.size, pad_value, dtype=np.float32)
    return np.concatenate([arr, padding])

  def _build_features(self, target_lataccel, current_lataccel, state, future_plan) -> np.ndarray:
    features = np.zeros(self.INPUT_SIZE, dtype=np.float32)
    features[0] = target_lataccel
    features[1] = current_lataccel
    features[2] = state.roll_lataccel
    features[3] = state.v_ego
    features[4] = state.a_ego

    past_actions = self.past_actions.copy()
    if self.control_steps < self.CONTEXT_LENGTH:
      past_actions[:self.CONTEXT_LENGTH - self.control_steps] = 0.0
    features[5:25] = past_actions
    features[25:45] = self.past_lataccels
    features[45:95] = self._pad_future(future_plan.lataccel, target_lataccel)
    features[95:145] = self._pad_future(future_plan.roll_lataccel, state.roll_lataccel)
    features[145:195] = self._pad_future(future_plan.v_ego, state.v_ego)
    features[195:245] = self._pad_future(future_plan.a_ego, state.a_ego)
    features[245] = self.control_steps / self.SEGMENT_LENGTH
    features[246] = min(self.control_steps, self.CONTEXT_LENGTH) / self.CONTEXT_LENGTH
    return features

  def _update_buffers(self, action: float, current_lataccel: float) -> None:
    self.past_actions = np.roll(self.past_actions, -1)
    self.past_actions[-1] = action
    self.past_lataccels = np.roll(self.past_lataccels, -1)
    self.past_lataccels[-1] = current_lataccel

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    if self.model is None:
      self._load_model()
    assert self.model is not None

    features = self._build_features(target_lataccel, current_lataccel, state, future_plan)
    with torch.no_grad():
      tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
      action = float(self.model(tensor).item())
    action = float(np.clip(np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0), -2.0, 2.0))

    self._update_buffers(action, current_lataccel)
    self.total_steps += 1
    if self._in_control:
      self.control_steps += 1
    return action
