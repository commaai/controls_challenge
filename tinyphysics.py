import argparse
import importlib
import numpy as np
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal

from collections import namedtuple
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import List, Union, Tuple
from tqdm.contrib.concurrent import process_map

from controllers import BaseController

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 550
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])


class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    return self.bins[token]

  def clip(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


class TinyPhysicsModel:
  def __init__(self, model_path: str, debug: bool) -> None:
    self.tokenizer = LataccelTokenizer()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    provider = 'CPUExecutionProvider'

    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, [provider])

  def softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def predict(self, input_data: dict, temperature=1.) -> int:
    res = self.ort_session.run(None, input_data)[0]
    probs = self.softmax(res / temperature, axis=-1)
    # we only care about the last timestep (batch size is just 1)
    assert probs.shape[0] == 1
    assert probs.shape[2] == VOCAB_SIZE
    sample = np.random.choice(probs.shape[2], p=probs[0, -1])
    return sample

  def get_current_lataccel(self, sim_states: List[State], actions: List[float], past_preds: List[float]) -> float:
    tokenized_actions = self.tokenizer.encode(past_preds)
    raw_states = [list(x) for x in sim_states]
    states = np.column_stack([actions, raw_states])
    input_data = {
      'states': np.expand_dims(states, axis=0).astype(np.float32),
      'tokens': np.expand_dims(tokenized_actions, axis=0).astype(np.int64)
    }
    return self.tokenizer.decode(self.predict(input_data, temperature=1.))


class TinyPhysicsSimulator:
  def __init__(self, model: TinyPhysicsModel, data_path: str, controller: BaseController, debug: bool = False) -> None:
    self.data_path = data_path
    self.sim_model = model
    self.data = self.get_data(data_path)
    self.controller = controller
    self.debug = debug
    self.reset()

  def reset(self) -> None:
    self.step_idx = CONTEXT_LENGTH
    state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
    self.state_history = [x[0] for x in state_target_futureplans]
    self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
    self.current_lataccel_history = [x[1] for x in state_target_futureplans]
    self.target_lataccel_history = [x[1] for x in state_target_futureplans]
    self.target_future = None
    self.current_lataccel = self.current_lataccel_history[-1]
    seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)

  def get_data(self, data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
      'v_ego': df['vEgo'].values,
      'a_ego': df['aEgo'].values,
      'target_lataccel': df['targetLateralAcceleration'].values,
      'steer_command': df['steerCommand'].values
    })
    return processed_df

  def sim_step(self, step_idx: int) -> None:
    pred = self.sim_model.get_current_lataccel(
      sim_states=self.state_history[-CONTEXT_LENGTH:],
      actions=self.action_history[-CONTEXT_LENGTH:],
      past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
    )
    pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    if step_idx >= CONTROL_START_IDX:
      self.current_lataccel = pred
    else:
      self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

    self.current_lataccel_history.append(self.current_lataccel)

  def control_step(self, step_idx: int) -> None:
    action = self.controller.update(self.target_lataccel_history[step_idx], self.current_lataccel, self.state_history[step_idx], future_plan=self.futureplan)
    if step_idx < CONTROL_START_IDX:
      action = self.data['steer_command'].values[step_idx]
    action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
    self.action_history.append(action)

  def get_state_target_futureplan(self, step_idx: int) -> Tuple[State, float]:
    state = self.data.iloc[step_idx]
    return (
      State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
      state['target_lataccel'],
      FuturePlan(
        lataccel=self.data['target_lataccel'].values[step_idx + 1 :step_idx + FUTURE_PLAN_STEPS].tolist(),
        roll_lataccel=self.data['roll_lataccel'].values[step_idx + 1 :step_idx + FUTURE_PLAN_STEPS].tolist(),
        v_ego=self.data['v_ego'].values[step_idx + 1 :step_idx + FUTURE_PLAN_STEPS].tolist(),
        a_ego=self.data['a_ego'].values[step_idx + 1 :step_idx + FUTURE_PLAN_STEPS].tolist()
      )
    )

  def step(self) -> None:
    state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
    self.state_history.append(state)
    self.target_lataccel_history.append(target)
    self.futureplan = futureplan
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

  def compute_cost(self) -> dict:
    target = np.array(self.target_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]
    pred = np.array(self.current_lataccel_history)[CONTROL_START_IDX:COST_END_IDX]

    lat_accel_cost = np.mean((target - pred)**2) * 100
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2) * 100
    total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
    return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

  def rollout(self) -> float:
    if self.debug:
      plt.ion()
      fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)

    for _ in range(CONTEXT_LENGTH, len(self.data)):
      self.step()
      if self.debug and self.step_idx % 10 == 0:
        print(f"Step {self.step_idx:<5}: Current lataccel: {self.current_lataccel:>6.2f}, Target lataccel: {self.target_lataccel_history[-1]:>6.2f}")
        self.plot_data(ax[0], [(self.target_lataccel_history, 'Target lataccel'), (self.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
        self.plot_data(ax[1], [(self.action_history, 'Action')], ['Step', 'Action'], 'Action')
        self.plot_data(ax[2], [(np.array(self.state_history)[:, 0], 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
        self.plot_data(ax[3], [(np.array(self.state_history)[:, 1], 'v_ego')], ['Step', 'v_ego'], 'v_ego')
        plt.pause(0.01)

    if self.debug:
      plt.ioff()
      plt.show()
    return self.compute_cost()


def get_available_controllers():
  return [f.stem for f in Path('controllers').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']


def run_rollout(data_path, controller_type, model_path, debug=False):
  tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
  controller = importlib.import_module(f'controllers.{controller_type}').Controller()
  sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_path), controller=controller, debug=debug)
  return sim.rollout(), sim.target_lataccel_history, sim.current_lataccel_history


if __name__ == "__main__":
  available_controllers = get_available_controllers()
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--debug", action='store_true')
  parser.add_argument("--controller", default='pid', choices=available_controllers)
  args = parser.parse_args()

  data_path = Path(args.data_path)
  if data_path.is_file():
    cost, _, _ = run_rollout(data_path, args.controller, args.model_path, debug=args.debug)
    print(f"\nAverage lataccel_cost: {cost['lataccel_cost']:>6.4}, average jerk_cost: {cost['jerk_cost']:>6.4}, average total_cost: {cost['total_cost']:>6.4}")
  elif data_path.is_dir():
    run_rollout_partial = partial(run_rollout, controller_type=args.controller, model_path=args.model_path, debug=False)
    files = sorted(data_path.iterdir())[:args.num_segs]
    results = process_map(run_rollout_partial, files, max_workers=16)
    costs = [result[0] for result in results]
    costs_df = pd.DataFrame(costs)
    print(f"\nAverage lataccel_cost: {np.mean(costs_df['lataccel_cost']):>6.4}, average jerk_cost: {np.mean(costs_df['jerk_cost']):>6.4}, average total_cost: {np.mean(costs_df['total_cost']):>6.4}")
    for cost in costs_df.columns:
      plt.hist(costs_df[cost], bins=np.arange(0, 1000, 10), label=cost, alpha=0.5)
    plt.xlabel('costs')
    plt.ylabel('Frequency')
    plt.title('costs Distribution')
    plt.legend()
    plt.show()
