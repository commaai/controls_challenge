@echo off
echo Starting PID Controller Tuning Process
echo ==================================

REM Install requirements
pip install -r requirements.txt

REM Activate environment if needed
REM call venv\Scripts\activate.bat

REM Run PID tuning with recommended parameters
python scripts/tune_pid.py ^
  --model_path ./models/tinyphysics.onnx ^
  --data_path ./data ^
  --num_segs 200 ^
  --n_trials 100 ^
  --update_controller

echo.
echo Tuning completed! Updated controller with best parameters.
echo Running final evaluation against baseline PID...
echo.

REM Generate final evaluation report
python eval.py ^
  --model_path ./models/tinyphysics.onnx ^
  --data_path ./data ^
  --num_segs 100 ^
  --test_controller tuned_pid ^
  --baseline_controller pid

echo.
echo Evaluation complete! See report.html for results.
echo ================================== 