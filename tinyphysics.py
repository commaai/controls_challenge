import argparse
import numpy as np
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal

from collections import namedtuple
from hashlib import md5
from pathlib import Path
from typing import List, Union, Tuple
from tqdm import tqdm

from controllers import BaseController, CONTROLLERS

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

ACC_G = 9.81
SIM_START_IDX = 100
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-4, 4]
STEER_RANGE = [-1, 1]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1

State = namedtuple('State', ['roll_lataccel', 'vEgo', 'aEgo'])


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
    if 'CUDAExecutionProvider' in ort.get_available_providers():
      if debug:
        print("ONNX Runtime is using GPU")
      provider = ('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT'})
    else:
      if debug:
        print("ONNX Runtime is using CPU")
      provider = 'CPUExecutionProvider'

    with open(model_path, "rb") as f:
      self.ort_session = ort.InferenceSession(f.read(), options, [provider])

  def softmax(self, x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

  def predict(self, input_data: dict, temperature=1.) -> dict:
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
  def __init__(self, model: TinyPhysicsModel, data_path: str, do_sim_step: bool, do_control_step: bool, controller: BaseController, debug: bool = False) -> None:
    self.data_path = data_path
    self.sim_model = model
    self.data = self.get_data(data_path)
    self.do_sim_step = do_sim_step
    self.do_control_step = do_control_step
    self.controller = controller
    self.debug = debug
    self.times = []
    self.reset()

  def reset(self) -> None:
    self.step_idx = 0
    self.state_history = []
    self.action_history = []
    self.current_lataccel_history = []
    self.target_lataccel_history = []
    self.current_lataccel = 0.
    seed = int(md5(self.data_path.encode()).hexdigest(), 16) % 10**4
    np.random.seed(seed)

  def get_data(self, data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
      'vEgo': df['vEgo'].values,
      'aEgo': df['aEgo'].values,
      'target_lataccel': df['latAccelSteeringAngle'].values,
    })
    return processed_df

  def sim_step(self, step_idx: int) -> None:
    if self.do_sim_step and step_idx >= SIM_START_IDX:
      pred = self.sim_model.get_current_lataccel(
        sim_states=self.state_history[max(0, step_idx - CONTEXT_LENGTH):step_idx],
        actions=self.action_history[max(0, step_idx - CONTEXT_LENGTH):step_idx],
        past_preds=self.current_lataccel_history[max(0, step_idx - CONTEXT_LENGTH):step_idx]
      )
      self.current_lataccel = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
    else:
      self.current_lataccel = self.data.iloc[step_idx]['target_lataccel']
    self.current_lataccel_history.append(self.current_lataccel)

  def control_step(self, step_idx: int) -> None:
    if self.do_control_step and step_idx >= SIM_START_IDX:
      action = self.controller.update(self.target_lataccel_history[step_idx], self.current_lataccel, self.state_history[step_idx])
    else:
      action = 0.
    action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
    self.action_history.append(action)

  def get_state_target(self, step_idx: int) -> Tuple[List, float]:
    state = self.data.iloc[step_idx]
    return State(roll_lataccel=state['roll_lataccel'], vEgo=state['vEgo'], aEgo=state['aEgo']), state['target_lataccel']

  def step(self) -> None:
    state, target = self.get_state_target(self.step_idx)
    self.state_history.append(state)
    self.target_lataccel_history.append(target)
    self.control_step(self.step_idx)
    self.sim_step(self.step_idx)
    self.step_idx += 1

  def plot_data(self, ax, lines, axis_labels, title) -> None:
    ax.clear()
    for line, label in lines:
      ax.plot(line, label=label)
    ax.legend()
    ax.set_title(f"{title} | Step: {self.step_idx}")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

  def compute_cost(self) -> float:
    target = np.array(self.target_lataccel_history)[SIM_START_IDX:]
    pred = np.array(self.current_lataccel_history)[SIM_START_IDX:]

    lat_accel_cost = np.mean((target - pred)**2)
    jerk_cost = np.mean((np.diff(pred) / DEL_T)**2)
    return lat_accel_cost, jerk_cost

  def rollout(self) -> None:
    if self.debug:
      plt.ion()
      fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)

    for _ in range(len(self.data)):
      self.step()
      if self.debug and self.step_idx % 10 == 0:
        print(f"Step {self.step_idx:<5}: Current lataccel: {self.current_lataccel:>6.2f}, Target lataccel: {self.target_lataccel_history[-1]:>6.2f}")
        self.plot_data(ax[0], [(self.target_lataccel_history, 'Target lataccel'), (self.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
        self.plot_data(ax[1], [(self.action_history, 'Action')], ['Step', 'Action'], 'Action')
        self.plot_data(ax[2], [(np.array(self.state_history)[:, 0], 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
        self.plot_data(ax[3], [(np.array(self.state_history)[:, 1], 'vEgo')], ['Step', 'vEgo'], 'vEgo')
        plt.pause(0.01)

    if self.debug:
      plt.ioff()
      plt.show()
    return self.compute_cost()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--do_sim_step", action='store_true')
  parser.add_argument("--do_control_step", action='store_true')
  parser.add_argument("--debug", action='store_true')
  parser.add_argument("--controller", default='simple', choices=CONTROLLERS.keys())
  args = parser.parse_args()

  tinyphysicsmodel = TinyPhysicsModel(args.model_path, debug=args.debug)
  controller = CONTROLLERS[args.controller]()

  data_path = Path(args.data_path)
  if data_path.is_file():
    sim = TinyPhysicsSimulator(tinyphysicsmodel, args.data_path, args.do_sim_step, args.do_control_step, controller=controller, debug=args.debug)
    lat_accel_cost, jerk_cost = sim.rollout()
    print(f"\nAverage lat_accel_cost: {lat_accel_cost:>6.4}, average jerk_cost: {jerk_cost:>6.4}")
  elif data_path.is_dir():
    costs = []
    files = sorted(data_path.iterdir())[:args.num_segs]
    for data_file in tqdm(files, total=len(files)):
      sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_file), args.do_sim_step, args.do_control_step, controller=controller, debug=args.debug)
      cost = sim.rollout()
      costs.append(cost)
    costs = np.array(costs)
    print(f"\nAverage lat_accel_cost: {np.mean(costs[:, 0]):>6.4}, average jerk_cost: {np.mean(costs[:, 1]):>6.4}")
    plt.hist(costs[:, 0], bins=np.arange(0, 2, 0.1), label='lat_accel_cost', alpha=0.5)
    plt.hist(costs[:, 1], bins=np.arange(0, 2, 0.1), label='jerk_cost', alpha=0.5)
    plt.xlabel('costs')
    plt.ylabel('Frequency')
    plt.title('costs Distribution')
    plt.show()
