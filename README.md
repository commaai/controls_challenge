# Comma Controls Challenge!

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.


## Get Started
We'll be using driving segments from the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge.

```
# download necessary dataset (~1.2G)
bash ./download_dataset.sh

# Test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --do_sim_step --do_control_step --vis

```
