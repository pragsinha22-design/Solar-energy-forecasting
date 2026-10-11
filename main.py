"""
main.py - run the whole project pipeline in order.

    python main.py                 # everything
    python main.py --skip-eda      # skip the EDA graphs
    python main.py --skip-arima    # skip ARIMA (it is the slowest step)

Each step is an ordinary script in src/ that you can also run on its own.
The pipeline stops immediately if a step fails.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Preprocessing (load, align time zones, merge)", "src/preprocessing.py", "preprocessing"),
    ("EDA graphs", "src/eda.py", "eda"),
    ("Feature engineering + 1-hour-ahead target", "src/features.py", "features"),
    ("Data-leakage checks", "src/leakage_check.py", "leakage"),
    ("Baseline + Linear Regression + XGBoost", "src/regression_models.py", "regression"),
    ("ARIMA (slow: rolling forecasts)", "src/time_series.py", "arima"),
    ("Evaluation, comparison table and graphs", "src/evaluation.py", "evaluation"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-arima", action="store_true")
    args = parser.parse_args()

    for title, script, key in STEPS:
        if (key == "eda" and args.skip_eda) or (key == "arima" and args.skip_arima):
            print(f"\n--- SKIPPED: {title}")
            continue
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nStep failed: {script}. Fix the error above and run again.")
            sys.exit(result.returncode)

    print("\nAll steps finished. Start the dashboard with:\n    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
