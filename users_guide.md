# Controls Challenge - Complete User Guide

## Overview

**Data**: Synthetic comma-steering-control dataset (real openpilot driving data)  
**Model**: TinyPhysics ML model simulates car lateral movement  
**Controllers**: Implement BaseController, output steering from target vs current lateral accel  
**Eval**: lataccel_cost + jerk_cost, use eval.py for reports  
**CSV**: Time-series with velocity, acceleration, steering, target lateral accel  

## Available Controllers

- **PID**: Baseline proportional controller
- **Q-Learning**: Tabular RL with discretized state space  
- **DQN**: Deep Q-Network with continuous states
- **PPO**: Proximal Policy Optimization with continuous actions

## Quick Start

**Training**: `python algorithms/sb3_ppo/train_sb3_ppo.py`  
**Evaluation**: `python eval.py --test_controller sb3_ppo_controller --baseline_controller pid`

## CSV Data Format
```csv
t,vEgo,aEgo,roll,targetLateralAcceleration,steerCommand
0.0,33.77,-0.017,0.037,1.004,-0.330
0.1,16.69,-0.071,0.023,0.017,0.115
```

---

# Controller Development Guide

## 🎯 CRITICAL FOR CONTROLLER: `controller.update()` Interface

Your controller must implement this signature:
```python
def update(self, target_lataccel: float, current_lataccel: float, state: State, future_plan: FuturePlan) -> float:
    # Return steering action in range [-2, 2]
```

**Action Range**: `[-2, 2]` (STEER_RANGE)

## 📊 Data Structures Your Controller Receives

### State
`namedtuple` with:
- `roll_lataccel`: Lateral acceleration from road banking (m/s²)
- `v_ego`: Vehicle speed (m/s)  
- `a_ego`: Vehicle acceleration (m/s²)

### FuturePlan
5-second future trajectory with:
- `lataccel`: Target lateral accelerations
- `roll_lataccel`: Future road banking effects
- `v_ego`: Future speeds
- `a_ego`: Future accelerations

## ⚡ Controller Integration Summary

**What you control**: Steering commands in `[-2, 2]`  
**What you receive**: Current state + 5-second future plan  
**Goal**: Minimize lateral acceleration tracking error + jerk  
**Evaluation**: Steps 100-500 (4 seconds of control)  

### Key Constants:
- Control starts at step 100
- 10 Hz update rate
- Max lataccel change: 0.5 m/s² per step
- Lateral accel range: [-5, 5] m/s²
- Future plan duration: 5 seconds

## 🔄 Simulation Flow

1. **Initialization**: Load data, reset histories
2. **For each timestep**:
   - Get current state and future plan
   - Call your `controller.update()`
   - Clip action to `[-2, 2]`
   - Use transformer to predict next lateral acceleration
   - Update histories
3. **Evaluation**: Calculate costs on steps 100-500

## 📈 Performance Metrics

- **lataccel_cost**: How well you track the target lateral acceleration
- **jerk_cost**: How smooth your control is (penalizes rapid changes)
- **total_cost**: `lataccel_cost × 50 + jerk_cost` (lateral tracking is heavily weighted)

## 💡 Controller Development Tips

1. **Use the future plan**: The 5-second lookahead gives you valuable trajectory information
2. **Balance tracking vs smoothness**: Large steering changes increase jerk cost
3. **Consider vehicle dynamics**: Speed and acceleration affect how steering translates to lateral acceleration
4. **Account for road banking**: `roll_lataccel` affects the vehicle's natural lateral acceleration

---

# Technical Implementation Details

## TinyPhysics Architecture Functions

### LataccelTokenizer Class
- `__init__()`: Creates 1024 bins from -5 to 5 m/s²
- `encode()`: Converts continuous lataccel → discrete token
- `decode()`: Converts token → continuous lataccel  
- `clip()`: Clamps values to [-5, 5] range

*Not directly used by controller*

### TinyPhysicsModel Class
- `__init__()`: Loads ONNX transformer model
- `softmax()`: Probability distribution calculation
- `predict()`: Gets next lataccel token from model
- `get_current_lataccel()`: Main model prediction interface

*Physics simulation - not for controller*

### TinyPhysicsSimulator Class

#### Key Functions:
- `__init__()`: Sets up simulation with your controller
- `reset()`: Initializes 20-step history buffer
- `get_data()`: Processes CSV data (converts steering convention)
- `control_step()`: **CALLS YOUR CONTROLLER**
- `sim_step()`: Updates physics using transformer model
- `step()`: Runs one simulation timestep
- `rollout()`: **MAIN EVALUATION LOOP**
- `compute_cost()`: Calculates performance metrics

#### Cost Evaluation:
- Steps 100-500 are evaluated
- `lataccel_cost`: MSE between target/actual × 100
- `jerk_cost`: Smoothness penalty × 100  
- `total_cost`: Weighted sum (lataccel × 50 + jerk)

### Utility Functions
- `get_available_controllers()`: Lists controller files
- `run_rollout()`: Single trajectory evaluation
- `download_dataset()`: Gets training data

Your controller's `update()` method is called once per timestep in `control_step()` and must return a float steering command.