#!/usr/bin/env python
"""
PID Controller Parameter Tuning with Optuna

This script uses Optuna to find optimal gains for the tuned_pid controller
by running the simulator on a subset of the data segments.

Usage:
    python scripts/tune_pid.py --model_path models/tinyphysics.onnx --data_path data --num_segs 200
"""

import argparse
import importlib
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

# Add parent directory to path to allow importing tinyphysics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# Import controller
from controllers.tuned_pid import Controller

def objective(trial, model_path, data_files, num_segs=200, lataccel_weight=50):
    """Optuna objective function that evaluates PID parameters"""
    # Sample hyperparameters - main gains
    p = trial.suggest_float("p", 0.05, 0.8, log=True)
    i = trial.suggest_float("i", 0.01, 0.3, log=True)
    d = trial.suggest_float("d", -0.5, 0.0, log=False)
    k_ff = trial.suggest_float("k_ff", 0.005, 0.2, log=True)
    
    # Setup the model
    tinyphysicsmodel = TinyPhysicsModel(model_path, debug=False)
    
    # Shuffle and select a subset of data files
    np.random.shuffle(data_files)
    selected_data_files = data_files[:num_segs]
    
    # Collect costs for each segment
    costs = []
    for i_seg, data_file in enumerate(selected_data_files):
        # Progress indicator
        if i_seg % 10 == 0:
            print(f"Progress: {i_seg}/{len(selected_data_files)} segments")
        
        # Create controller with the trial parameters
        controller = Controller()
        # Update controller parameters
        controller.p = p
        controller.i = i
        controller.d = d
        controller.k_ff = k_ff
        
        # Create simulator and run rollout
        sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_file), controller=controller, debug=False)
        cost = sim.rollout()
        costs.append(cost)
    
    # Calculate the mean costs across all segments
    if costs:
        lataccel_costs = np.mean([c['lataccel_cost'] for c in costs])
        jerk_costs = np.mean([c['jerk_cost'] for c in costs]) 
        total_cost = np.mean([c['total_cost'] for c in costs])
        
        # We can weight different aspects differently during tuning
        # tuning_cost = lataccel_weight * lataccel_costs + jerk_costs
        tuning_cost = total_cost  # Default to using simulator's total_cost
    else:
        tuning_cost = float('inf')  # If no valid costs, this is a bad trial
    
    print(f"Trial {trial.number}: p={p:.4f}, i={i:.4f}, d={d:.4f}, k_ff={k_ff:.4f}, " +
          f"cost={tuning_cost:.4f}")
    
    return tuning_cost

def run_tuning(model_path, data_path, num_segs=200, n_trials=100, lataccel_weight=50):
    """Run the Optuna tuning process"""
    # Get data files
    data_files = list(Path(data_path).glob('*.csv'))
    if not data_files:
        raise ValueError(f"No data files found in {data_path}")
    
    print(f"Found {len(data_files)} data files, will use up to {num_segs} for each trial")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    # Run optimization
    objective_func = partial(objective, model_path=model_path, 
                           data_files=data_files, num_segs=num_segs,
                           lataccel_weight=lataccel_weight)
                           
    study.optimize(objective_func, n_trials=n_trials)
    
    # Print results
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n" + "="*50)
    print("Best Parameters:")
    print(f"p: {best_params['p']:.6f}")
    print(f"i: {best_params['i']:.6f}")
    print(f"d: {best_params['d']:.6f}")
    print(f"k_ff: {best_params['k_ff']:.6f}")
    print(f"Best Total Cost: {best_value:.4f}")
    print("="*50)
    
    # Save best parameters to a file
    with open("best_pid_params.txt", "w") as f:
        f.write(f"p = {best_params['p']:.6f}\n")
        f.write(f"i = {best_params['i']:.6f}\n")
        f.write(f"d = {best_params['d']:.6f}\n")
        f.write(f"k_ff = {best_params['k_ff']:.6f}\n")
        f.write(f"# Best Total Cost: {best_value:.4f}\n")
    
    # Try to generate visualization if matplotlib is available
    try:
        # Create visualizations directory
        os.makedirs("visualizations", exist_ok=True)
        
        # Save parameter importance plot
        param_importance_fig = plot_param_importances(study)
        param_importance_fig.write_image("visualizations/param_importance.png")
        
        # Save optimization history plot
        history_fig = plot_optimization_history(study)
        history_fig.write_image("visualizations/optimization_history.png")
        
        print("Saved visualization plots to 'visualizations' directory")
    except Exception as e:
        print(f"Could not generate visualizations: {e}")
    
    return best_params, best_value

def modify_tuned_pid_with_best_params(best_params):
    """Update the tuned_pid.py file with the best parameters"""
    controller_path = Path("controllers/tuned_pid.py")
    with open(controller_path, "r") as f:
        content = f.read()
    
    # Update the parameters in the __init__ method
    # Find line with self.p assignment and update it
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "self.p =" in line:
            lines[i] = f"    self.p = {best_params['p']:.6f}"
        elif "self.i =" in line:
            lines[i] = f"    self.i = {best_params['i']:.6f}"
        elif "self.d =" in line:
            lines[i] = f"    self.d = {best_params['d']:.6f}"
        elif "self.k_ff =" in line:
            lines[i] = f"    self.k_ff = {best_params['k_ff']:.6f}"
    
    with open(controller_path, "w") as f:
        f.write('\n'.join(lines))
    
    print(f"Updated {controller_path} with the best parameters")

def main():
    parser = argparse.ArgumentParser(description="Tune PID controller parameters using Optuna")
    parser.add_argument("--model_path", required=True, help="Path to the model file")
    parser.add_argument("--data_path", required=True, help="Path to the data directory")
    parser.add_argument("--num_segs", type=int, default=200, help="Number of segments to use for tuning")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--lataccel_weight", type=float, default=50, 
                        help="Weight for lataccel_cost in the objective function")
    parser.add_argument("--update_controller", action="store_true", help="Update the controller file with best params")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Make sure optuna is installed
    try:
        import optuna
    except ImportError:
        print("Optuna not found. Installing...")
        os.system("pip install optuna")
    
    # Also ensure plotly is available for visualizations
    try:
        import plotly
    except ImportError:
        print("Plotly not found. Installing...")
        os.system("pip install plotly kaleido")
    
    best_params, best_value = run_tuning(
        model_path=args.model_path,
        data_path=args.data_path,
        num_segs=args.num_segs,
        n_trials=args.n_trials,
        lataccel_weight=args.lataccel_weight
    )
    
    # Optionally update the controller file
    if args.update_controller:
        modify_tuned_pid_with_best_params(best_params)
    
    elapsed_time = time.time() - start_time
    print(f"Tuning completed in {elapsed_time/60:.2f} minutes")

if __name__ == "__main__":
    main() 