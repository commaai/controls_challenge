import argparse
import base64
import numpy as np
import pandas as pd
import seaborn as sns


from functools import partial
from io import BytesIO
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tinyphysics import CONTROL_START_IDX, get_available_controllers, run_rollout

sns.set_theme()
SAMPLE_ROLLOUTS = 5

COLORS = {
  'test': '#c0392b',
  'baseline': '#2980b9'
}


def img2base64(fig):
  buf = BytesIO()
  fig.savefig(buf, format='png')
  data = base64.b64encode(buf.getbuffer()).decode("ascii")
  return data


def create_report(test, baseline, sample_rollouts, costs, num_segs):
  res = []
  res.append("""
  <html>
    <head>
    <title>Comma Controls Challenge: Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap" rel="stylesheet">
    <style type='text/css'>
      table {border-collapse: collapse; font-size: 30px; margin-top: 20px; margin-bottom: 30px;}
      th, td {border: 1px solid black; text-align: left; padding: 20px;}
      th {background-color: #f2f2f2;}
      th {background-color: #f2f2f2;}
    </style>
    </head>
    <body style="font-family: 'JetBrains Mono', monospace; margin: 20px; padding: 20px; display: flex; flex-direction: column; justify-content: center; align-items: center">
    """)
  res.append("<h1 style='font-size: 50px; font-weight: 700; text-align: center'>Comma Controls Challenge: Report</h1>")
  res.append(f"<h3 style='font-size: 30px;'><span style='background: {COLORS['test']}; color: #fff; padding: 10px'>Test Controller: {test}</span> | <span style='background: {COLORS['baseline']}; color: #fff; padding: 10px'>Baseline Controller: {baseline}</span></h3>")

  res.append(f"<h2 style='font-size: 30px; margin-top: 50px'>Aggregate Costs (total rollouts: {num_segs})</h2>")
  res_df = pd.DataFrame(costs)
  fig, axs = plt.subplots(ncols=3, figsize=(18, 6), sharey=True)
  bins = np.arange(0, 1000, 10)
  for ax, cost in zip(axs, ['lataccel_cost', 'jerk_cost', 'total_cost']):
    for controller in ['test', 'baseline']:
      ax.hist(res_df[res_df['controller'] == controller][cost], bins=bins, label=controller, alpha=0.5, color=COLORS[controller])
    ax.set_xlabel('Cost')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Cost Distribution: {cost}')
    ax.legend()
  res.append(f'<img style="max-width:100%" src="data:image/png;base64,{img2base64(fig)}" alt="Plot">')
  agg_df = res_df.groupby('controller').agg({'lataccel_cost': 'mean', 'jerk_cost': 'mean', 'total_cost': 'mean'}).round(3).reset_index()
  res.append(agg_df.to_html(index=False))

  passed_baseline = agg_df[agg_df['controller'] == 'test']['total_cost'].values[0] < agg_df[agg_df['controller'] == 'baseline']['total_cost'].values[0]
  if passed_baseline:
    res.append(f"<h3 style='font-size: 20px; color: #27ae60'> ✅ Test Controller ({test}) passed Baseline Controller ({baseline})! ✅ </h3>")
    res.append("""<p>Check the leaderboard
      <a href='https://comma.ai/leaderboard'>here</a>
      and submit your results
      <a href='https://docs.google.com/forms/d/e/1FAIpQLSc_Qsh5egoseXKr8vI2TIlsskd6nNZLNVuMJBjkogZzLe79KQ/viewform'>here</a>
      !</p>""")
  else:
    res.append(f"<h3 style='font-size: 20px; color: #c0392b'> ❌ Test Controller ({test}) failed to beat Baseline Controller ({baseline})! ❌</h3>")

  res.append("<hr style='border: #ddd 1px solid; width: 80%'>")
  res.append("<h2  style='font-size: 30px; margin-top: 50px'>Sample Rollouts</h2>")
  fig, axs = plt.subplots(ncols=1, nrows=SAMPLE_ROLLOUTS, figsize=(15, 3 * SAMPLE_ROLLOUTS), sharex=True)
  for ax, rollout in zip(axs, sample_rollouts):
    ax.plot(rollout['desired_lataccel'], label='Desired Lateral Acceleration', color='#27ae60')
    ax.plot(rollout['test_controller_lataccel'], label='Test Controller Lateral Acceleration', color=COLORS['test'])
    ax.plot(rollout['baseline_controller_lataccel'], label='Baseline Controller Lateral Acceleration', color=COLORS['baseline'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Lateral Acceleration')
    ax.set_title(f"Segment: {rollout['seg']}")
    ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
    ax.legend()
  fig.tight_layout()
  res.append(f'<img style="max-width:100%" src="data:image/png;base64,{img2base64(fig)}" alt="Plot">')
  res.append("</body></html>")

  with open("report.html", "w") as fob:
    fob.write("\n".join(res))
    print("Report saved to: './report.html'")


if __name__ == "__main__":
  available_controllers = get_available_controllers()
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--data_path", type=str, required=True)
  parser.add_argument("--num_segs", type=int, default=100)
  parser.add_argument("--test_controller", default='pid', choices=available_controllers)
  parser.add_argument("--baseline_controller", default='pid', choices=available_controllers)
  args = parser.parse_args()

  data_path = Path(args.data_path)
  assert data_path.is_dir(), "data_path should be a directory"

  costs = []
  sample_rollouts = []
  files = sorted(data_path.iterdir())[:args.num_segs]
  print("Running rollouts for visualizations...")
  for d, data_file in enumerate(tqdm(files[:SAMPLE_ROLLOUTS], total=SAMPLE_ROLLOUTS)):
    test_cost, test_target_lataccel, test_current_lataccel = run_rollout(data_file, args.test_controller, args.model_path, debug=False)
    baseline_cost, baseline_target_lataccel, baseline_current_lataccel = run_rollout(data_file, args.baseline_controller, args.model_path, debug=False)
    sample_rollouts.append({
      'seg': data_file.stem,
      'test_controller': args.test_controller,
      'baseline_controller': args.baseline_controller,
      'desired_lataccel': test_target_lataccel,
      'test_controller_lataccel': test_current_lataccel,
      'baseline_controller_lataccel': baseline_current_lataccel,
    })

    costs.append({'controller': 'test', **test_cost})
    costs.append({'controller': 'baseline', **baseline_cost})

  for controller_cat, controller_type in [('baseline', args.baseline_controller), ('test', args.test_controller)]:
    print(f"Running batch rollouts => {controller_cat} controller: {controller_type}")
    rollout_partial = partial(run_rollout, controller_type=controller_type, model_path=args.model_path, debug=False)
    results = process_map(rollout_partial, files[SAMPLE_ROLLOUTS:], max_workers=16, chunksize=10)
    costs += [{'controller': controller_cat, **result[0]} for result in results]

  create_report(args.test_controller, args.baseline_controller, sample_rollouts, costs, len(files))
