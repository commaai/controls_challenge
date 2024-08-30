import importlib
import numpy as np
import onnxruntime as ort
import pandas as pd

from collections import namedtuple
from pathlib import Path
from typing import List, Union, Tuple, Dict, Callable
from tqdm import tqdm
import random

from controllers import BaseController


ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0
MAX_BATCH_SIZE=1024
MAX_EPISODE_SIZE=450  # Force fixed length episodes

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])


class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value: Union[float, np.ndarray, List[float]]) -> np.ndarray:
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token: np.ndarray) -> np.ndarray:
    return self.bins[token]

  def clip(self, value: Union[float, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
  def __init__(self, model_path: str, debug: bool) -> None:
    self.tokenizer = LataccelTokenizer()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    provider = 'CUDAExecutionProvider'

    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, [provider])

  def softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def predict(self, input_data: dict, rng: np.random.Generator, temperature=1.) -> np.ndarray:
    res = self.ort_session.run(None, input_data)[0]

    # we only care about the last timestep
    probs = self.softmax(res[:, -1] / temperature, axis=-1)
    assert probs.shape[0] <= MAX_BATCH_SIZE
    assert probs.shape[1] == VOCAB_SIZE
    samples = (probs.cumsum(axis=1) > rng.random(probs.shape[0])[:, np.newaxis]).argmax(axis=1) # Inverse transform sampling
    return samples

  def get_current_lataccel(self, sim_states: np.ndarray, actions: np.ndarray, past_preds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    tokenized_actions = self.tokenizer.encode(past_preds)
    batch_states = np.concatenate((actions[:, :, np.newaxis], sim_states), axis=2)

    input_data = {
      'states': batch_states.astype(np.float32),
      'tokens': tokenized_actions.astype(np.int64)
    }
    return self.tokenizer.decode(self.predict(input_data, rng, temperature=0.8))


class TinyPhysicsSimulator:
  def __init__(self, model: TinyPhysicsModel, dfs: List[pd.DataFrame], controller: BaseController, episode_len: int=MAX_EPISODE_SIZE) -> None:
    self.batch_size = len(dfs)
    self.terminate_step = CONTROL_START_IDX + episode_len
    self.sim_model = model
    self.get_data(dfs)
    self.controller = controller
    self.reset()

  def reset(self) -> None:
    self.step_idx = CONTEXT_LENGTH
    self.current_lataccel_histories = self.target_lataccel_histories.copy()
    self.current_lataccel = self.current_lataccel_histories[:, self.step_idx - 1]
    self.futureplan = None
    self.rng = np.random.default_rng()

  def get_data(self, dfs: List[pd.DataFrame]) -> None:
    self.state_histories = np.array([np.column_stack([np.sin(d['roll'].to_numpy())[:self.terminate_step + FUTURE_PLAN_STEPS] * ACC_G,
                                    d['vEgo'].to_numpy()[:self.terminate_step + FUTURE_PLAN_STEPS],
                                    d['aEgo'].to_numpy()[:self.terminate_step + FUTURE_PLAN_STEPS]]) for d in dfs])
    self.target_lataccel_histories = np.array([d['targetLateralAcceleration'].to_numpy()[:self.terminate_step + FUTURE_PLAN_STEPS] for d in dfs])
    self.action_histories = np.array([-d['steerCommand'].to_numpy()[:self.terminate_step + FUTURE_PLAN_STEPS] for d in dfs]) # steer commands are logged with left-positive convention but this simulator uses right-positive

  def sim_step(self, step_idx: int) -> None:
    preds = self.sim_model.get_current_lataccel(
      sim_states=self.state_histories[:, step_idx - CONTEXT_LENGTH + 1: step_idx + 1],
      actions=self.action_histories[:, step_idx - CONTEXT_LENGTH + 1: step_idx + 1],
      past_preds=self.current_lataccel_histories[:, step_idx - CONTEXT_LENGTH: step_idx],
      rng=self.rng
    )
    preds = np.clip(preds, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    if step_idx >= CONTROL_START_IDX:
      self.current_lataccel = preds
    else:
      self.current_lataccel = self.target_lataccel_histories[:, step_idx]

    self.current_lataccel_histories[:, step_idx] = self.current_lataccel

  def control_step(self, step_idx: int) -> None:
    actions = self.controller.update(self.target_lataccel_histories[:, step_idx], self.current_lataccel, self.state_histories[:, step_idx], future_plan=self.futureplan)
    if step_idx < CONTROL_START_IDX:
      actions = self.action_histories[:, step_idx]

    actions = np.clip(actions, STEER_RANGE[0], STEER_RANGE[1])
    self.action_histories[:, step_idx] = actions

  def get_future_plan(self, step_idx: int) -> np.ndarray:
    return np.concatenate([self.state_histories[:, step_idx + 1: step_idx + FUTURE_PLAN_STEPS],
                           self.target_lataccel_histories[:, step_idx + 1: step_idx + FUTURE_PLAN_STEPS, np.newaxis]], axis=2)

  def step(self) -> None:
    self.futureplan = self.get_future_plan(self.step_idx)
    self.control_step(self.step_idx)
    self.sim_step(self.step_idx)
    self.step_idx += 1

  def compute_cost(self) -> Tuple[float, float, float]:
    target = self.target_lataccel_histories[:, CONTROL_START_IDX:]
    pred = self.current_lataccel_histories[:, CONTROL_START_IDX:]

    lat_accel_costs = np.mean((target - pred)**2, axis=1) * 100
    jerk_costs = np.mean((np.diff(pred, axis=1) / DEL_T)**2, axis=1) * 100
    total_costs = (lat_accel_costs * LAT_ACCEL_COST_MULTIPLIER) + jerk_costs

    return lat_accel_costs.mean(), jerk_costs.mean(), total_costs.mean()

  def rollout(self) -> Tuple[float, float, float]:
    for _ in range(CONTEXT_LENGTH, self.terminate_step):
      self.step()
    self.target_lataccel_histories = self.target_lataccel_histories[:, :self.terminate_step]
    self.current_lataccel_histories = self.current_lataccel_histories[:, :self.terminate_step]
    self.action_histories = self.action_histories[:, :self.terminate_step]
    return self.compute_cost()

def get_available_controllers():
  return [f.stem for f in Path('controllers').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']

def run_rollout(data_paths, controller_type, model_path, debug=False):
  if not isinstance(data_paths, list):
    data_paths = [data_paths]
  tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
  costs = []
  for i in tqdm(range(0, len(data_paths), MAX_BATCH_SIZE)):
    controller = importlib.import_module(f'controllers.{controller_type}').Controller()
    sim = TinyPhysicsSimulator(tinyphysicsmodel, data_paths[i: min(i + MAX_BATCH_SIZE, len(data_paths))], controller=controller)
    costs += sim.rollout()
  return costs

# if __name__ == "__main__":
#   available_controllers = get_available_controllers()
#   parser = argparse.ArgumentParser()
#   parser.add_argument("--model_path", type=str, required=True)
#   parser.add_argument("--data_path", type=str, required=True)
#   parser.add_argument("--num_segs", type=int, default=100)
#   parser.add_argument("--controller", default='pid', choices=available_controllers)
#   args = parser.parse_args()

#   data_path = Path(args.data_path)
#   if data_path.is_file():
#     cost, _, _ = run_rollout(data_path, args.controller, args.model_path)
#     print(f"\nAverage lataccel_cost: {cost[0]['lataccel_cost']:>6.4}, average jerk_cost: {cost[0]['jerk_cost']:>6.4}, average total_cost: {cost[0]['total_cost']:>6.4}")
#   elif data_path.is_dir():
#     files = sorted(data_path.iterdir())[:args.num_segs]
#     costs = run_rollout([str(f) for f in files], args.controller, args.model_path)
#     costs_df = pd.DataFrame(costs)
#     print(f"\nAverage lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4}, average total_cost: {np.mean(costs_df['total_cost']):>6.4}")