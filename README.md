# Comma Controls Challenge!
![Car](./imgs/car.jpg)

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.


## Geting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual routes with actual car and road states.

```
# download necessary dataset (~0.6G)
bash ./download_dataset.sh

# Test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller simple


# Batch Metrics on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller simple

# Generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller simple --baseline_controller zero

```


## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. It's inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`) and a steer input (`steer_action`) and predicts the resultant lateral acceleration fo the car.


## Controllers
Your controller should implement an [update function](https://github.com/commaai/controls_challenge/blob/1a25ee200f5466cb7dc1ab0bf6b7d0c67a2481db/controllers.py#L2) that returns the `steer_action`. This controller is then run in-loop, in the simulator to autoregressively predict the car's response.


## Evaluation
Each rollout will result in 2 costs:
- `lat_accel_cost`: $\dfrac{\Sigma(actual\_lat\_accel - target\_lat\_accel)^2}{steps}$

- `jerk_cost`: $\dfrac{\Sigma((actual\_lat\_accel_{t} - actual\_lat\_accel_{t-1}) / \Delta t)^2}{steps - 1}$


Minimizing both costs are very important.
