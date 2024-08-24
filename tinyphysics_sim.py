import argparse
import importlib
import numpy as np
import onnxruntime as ort
import pandas as pd

from collections import namedtuple
from hashlib import md5
from pathlib import Path
from typing import List, Union, Tuple, Dict
from tqdm import tqdm

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
MAX_BATCH_SIZE=256

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
    # options.intra_op_num_threads = 1
    # options.inter_op_num_threads = 1
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

  def get_current_lataccel(self, batch_sim_states: List[List[State]], batch_actions: List[List[float]], batch_past_preds: List[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    batch_tokenized_actions = self.tokenizer.encode(np.array(batch_past_preds)).T  # (CONTEXT_LENGTH, BATCH_SIZE)
    batch_raw_states = [[list(x) for x in sim_states] for sim_states in batch_sim_states] # (CONTEXT_LENGTH, BATCH_SIZE, 3)

    batch_actions = np.transpose(batch_actions)
    batch_raw_states = np.transpose(batch_raw_states, (1, 0, 2))

    batch_states = np.concatenate((batch_actions[:, :, np.newaxis], batch_raw_states), axis=2)
    # batch_states = [np.column_stack([actions, raw_states]) for actions, raw_states in zip(batch_actions, batch_raw_states)]

    input_data = {
      'states': np.array(batch_states).astype(np.float32),
      'tokens': batch_tokenized_actions.astype(np.int64)
    }
    return self.tokenizer.decode(self.predict(input_data, rng, temperature=0.8))


class TinyPhysicsSimulator:
  def __init__(self, model: TinyPhysicsModel, data_paths: Union[str, List[str]], controllers: List[BaseController]) -> None:
    if isinstance(data_paths, str):
      data_paths = [data_paths]
    if len(data_paths) > MAX_BATCH_SIZE:
        raise ValueError(f"batch size must be smaller than {MAX_BATCH_SIZE}")
    self.data_paths = data_paths
    self.sim_model = model
    self.data = list(map(self.get_data, data_paths))
    self.controllers = controllers
    self.is_batch_control = isinstance(controllers, BaseController) and len(data_paths) > 1
    self.reset()

  def reset(self) -> None:
    self.step_idx = CONTEXT_LENGTH
    batch_state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
    self.state_histories = [[x[0] for x in e] for e in batch_state_target_futureplans]
    self.action_histories = np.transpose([d['steer_command'][:self.step_idx] for d in self.data]).tolist()
    self.current_lataccel_histories = [np.array([x[1] for x in e]) for e in batch_state_target_futureplans]
    self.target_lataccel_histories = [np.array([x[1] for x in e]) for e in batch_state_target_futureplans]
    self.target_future = None
    self.current_lataccel = self.current_lataccel_histories[-1]
    self.rng = np.random.default_rng()

  def get_data(self, data_path: str) -> Dict[str, np.ndarray]:
    df = pd.read_csv(data_path)
    data = {
      'roll_lataccel': np.sin(df['roll'].to_numpy()) * ACC_G,
      'v_ego': df['vEgo'].to_numpy(),
      'a_ego': df['aEgo'].to_numpy(),
      'target_lataccel': df['targetLateralAcceleration'].to_numpy(),
      'steer_command': -df['steerCommand'].to_numpy()  # steer commands are logged with left-positive convention but this simulator uses right-positive
    }
    return data

  def sim_step(self, step_idx: int) -> None:
    preds = self.sim_model.get_current_lataccel(
      batch_sim_states=self.state_histories[-CONTEXT_LENGTH:],
      batch_actions=self.action_histories[-CONTEXT_LENGTH:],
      batch_past_preds=self.current_lataccel_histories[-CONTEXT_LENGTH:],
      random_generators=self.random_generators
    )
    preds = np.clip(preds, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    if step_idx >= CONTROL_START_IDX:
      self.current_lataccel = preds
    else:
      self.current_lataccel = self.target_lataccel_histories[-1]

    self.current_lataccel_histories.append(self.current_lataccel)

  def control_step(self, step_idx: int) -> None:
    actions = [controller.update(self.target_lataccel_histories[step_idx][i], self.current_lataccel[i], self.state_histories[step_idx][i], future_plan=self.futureplan[i]) for i, controller in enumerate(self.controllers)]
    if step_idx < CONTROL_START_IDX:
      actions = [d['steer_command'][step_idx] for d in self.data]
    actions = np.clip(actions, STEER_RANGE[0], STEER_RANGE[1]).tolist()
    self.action_histories.append(actions)

  def get_state_target_futureplan(self, step_idx: int) -> List[Tuple[State, float, FuturePlan]]:
    return [(
      State(roll_lataccel=d['roll_lataccel'][step_idx], v_ego=d['v_ego'][step_idx], a_ego=d['a_ego'][step_idx]),
      d['target_lataccel'][step_idx],
      FuturePlan(
        lataccel=d['target_lataccel'][step_idx + 1:step_idx + FUTURE_PLAN_STEPS],
        roll_lataccel=d['roll_lataccel'][step_idx + 1:step_idx + FUTURE_PLAN_STEPS],
        v_ego=d['v_ego'][step_idx + 1:step_idx + FUTURE_PLAN_STEPS],
        a_ego=d['a_ego'][step_idx + 1:step_idx + FUTURE_PLAN_STEPS]
      )
    ) for d in self.data]

  def step(self) -> None:
    batch_state_target_futureplan = self.get_state_target_futureplan(self.step_idx)
    self.state_histories.append([x[0] for x in batch_state_target_futureplan])
    self.target_lataccel_histories.append(np.array([x[1] for x in batch_state_target_futureplan]))
    self.futureplan = [x[2] for x in batch_state_target_futureplan]
    self.control_step(self.step_idx)
    self.sim_step(self.step_idx)
    self.step_idx += 1

  def plot_data(self, ax, lines, axis_labels, title) -> None:
    ax.clear()
    for line, label in lines:
      ax.plot(line, label=label)
    ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax.legend()
    ax.set_title(f"{title} | Step: {self.step_idx}")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

  def compute_cost(self) -> List[Dict[str, float]]:
    target = np.array(self.target_lataccel_histories)[CONTROL_START_IDX:COST_END_IDX]
    pred = np.array(self.current_lataccel_histories)[CONTROL_START_IDX:COST_END_IDX]

    lat_accel_costs = np.mean((target - pred)**2, axis=0) * 100
    jerk_costs = np.mean((np.diff(pred, axis=0) / DEL_T)**2, axis=0) * 100
    total_costs = (lat_accel_costs * LAT_ACCEL_COST_MULTIPLIER) + jerk_costs
    return [{'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost} for lat_accel_cost, jerk_cost, total_cost in zip(lat_accel_costs, jerk_costs, total_costs)]

  def rollout(self) -> List[Dict[str, float]]:
    for _ in range(CONTEXT_LENGTH, 550):
      self.step()

    return self.compute_cost()


def get_available_controllers():
  return [f.stem for f in Path('controllers').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']


def run_rollout(data_paths, controller_type, model_path, debug=False):
  if not isinstance(data_paths, list):
    data_paths = [data_paths]
  tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
  controllers = [importlib.import_module(f'controllers.{controller_type}').Controller() for _ in range(len(data_paths))]
  sim = TinyPhysicsSimulator(tinyphysicsmodel, [str(x) for x in data_paths], controllers=controllers)
  return sim.rollout(), sim.target_lataccel_histories, sim.current_lataccel_histories


if __name__ == "__main__":
  available_controllers = get_available_controllers()
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--controller", default='pid', choices=available_controllers)
  args = parser.parse_args()

  data_path = Path(args.data_path)
  if data_path.is_file():
    cost, _, _ = run_rollout(data_path, args.controller, args.model_path)
    print(f"\nAverage lataccel_cost: {cost[0]['lataccel_cost']:>6.4}, average jerk_cost: {cost[0]['jerk_cost']:>6.4}, average total_cost: {cost[0]['total_cost']:>6.4}")
  elif data_path.is_dir():
    files = sorted(data_path.iterdir())[:args.num_segs]
    costs = []
    for i in tqdm(range(0, args.num_segs, MAX_BATCH_SIZE)):
      costs += run_rollout(files[i: min(i + MAX_BATCH_SIZE, args.num_segs)], args.controller, args.model_path)[0]
    costs_df = pd.DataFrame(costs)
    print(f"\nAverage lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4}, average total_cost: {np.mean(costs_df['total_cost']):>6.4}")