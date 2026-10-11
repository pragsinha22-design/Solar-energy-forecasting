"""
leakage_check.py
================
Automatic tests that the features contain no future information.

Run from the project root (after features.py):
    python src/leakage_check.py

TEST LOGIC IN PLAIN WORDS
-------------------------
1. Lookup test      : re-compute every lag and the target a second, completely
                      different way (looking values up by timestamp) and compare.
2. Future-change test (the strongest one):
                      Pick a cut-off time. Replace ALL data after the cut-off
                      with random garbage and rebuild the features. If any
                      feature in a row BEFORE the cut-off changes, that feature
                      was using future data. (Only the target is allowed to
                      change, because it is the future by definition.)
3. Test-of-the-test : Do the same future-change test on a deliberately BAD
                      feature (shift(-1)) to prove the test can catch leakage.
4. Split test      : blocks are in time order, with no overlap even counting
                      the 1-hour-ahead answers.
5. Imputation test : no row used for modelling contains an interpolated value.
6. Target test     : the target is not among the inputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from features import (build_features, FEATURE_COLUMNS, TARGET_COLUMN, WEATHER_COLUMNS,
                      BASE_COLUMNS, MERGED_FILE, FLAGS_FILE, OUT_FILE, PROJECT_ROOT)

REPORT_FILE = PROJECT_ROOT / "results" / "leakage_check_report.txt"
lines = []
results = []


def log(text=""):
    print(text)
    lines.append(str(text))


def check(name, passed, detail=""):
    results.append(passed)
    log(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


raw = pd.read_csv(MERGED_FILE, parse_dates=["timestamp"]).set_index("timestamp")
flags = pd.read_csv(FLAGS_FILE, parse_dates=["timestamp"]).set_index("timestamp")
model = pd.read_csv(OUT_FILE, parse_dates=["timestamp"]).set_index("timestamp")
power = raw["ac_power"]  # original power series, looked up by timestamp

# ----------------------------------------------------------------------
# TEST 1: independent lookup of lags and target
# ----------------------------------------------------------------------
log("TEST 1: lags and target re-computed by timestamp lookup")
step = pd.Timedelta(minutes=15)
for col, offset in [("power_lag_1", -1 * step), ("power_lag_2", -2 * step),
                    ("power_lag_4", -4 * step), (TARGET_COLUMN, pd.Timedelta(hours=1)),
                    ("ac_power", pd.Timedelta(0))]:
    expected = power.reindex(model.index + offset).to_numpy()
    wrong = int((~np.isclose(model[col].to_numpy(), expected, equal_nan=False)).sum())
    check(f"{col} equals power at (t {'+' if offset >= pd.Timedelta(0) else '-'} "
          f"{abs(offset)})", wrong == 0, f"{wrong} mismatching rows out of {len(model)}")

# ----------------------------------------------------------------------
# TEST 2: future-change test
# ----------------------------------------------------------------------
log("\nTEST 2: replace everything after a cut-off with random numbers")
cutoff = pd.Timestamp("2025-06-01 00:00:00", tz="UTC")
rng = np.random.default_rng(0)
corrupted = raw[BASE_COLUMNS].copy()
future = corrupted.index > cutoff
corrupted.loc[future, :] = rng.uniform(0, 1000, size=(int(future.sum()), len(BASE_COLUMNS)))

original_feats = build_features(raw)
corrupted_feats = build_features(corrupted)
before = original_feats.index <= cutoff
a = original_feats.loc[before, FEATURE_COLUMNS].to_numpy(dtype=float)
b = corrupted_feats.loc[before, FEATURE_COLUMNS].to_numpy(dtype=float)
same = np.allclose(a, b, equal_nan=True)
check("All input features at times <= cut-off are unchanged by future data", same,
      f"{int(before.sum())} rows compared, {len(FEATURE_COLUMNS)} features each")
t_a = original_feats.loc[before, TARGET_COLUMN].to_numpy(dtype=float)
t_b = corrupted_feats.loc[before, TARGET_COLUMN].to_numpy(dtype=float)
n_target_changed = int((~np.isclose(t_a, t_b, equal_nan=True)).sum())
check("Target does change for the last 4 rows before cut-off (expected: it is the future)",
      n_target_changed == 4, f"{n_target_changed} target values changed")

# ----------------------------------------------------------------------
# TEST 3: can the test detect leakage? Try a deliberately leaky feature.
# ----------------------------------------------------------------------
log("\nTEST 3: sanity check - the test must catch a deliberately leaky feature")
orig_leaky = original_feats["ac_power"].shift(-1)       # NEXT step: future!
corr_leaky = corrupted_feats["ac_power"].shift(-1)
leak_changed = not np.allclose(orig_leaky[before].to_numpy(dtype=float),
                               corr_leaky[before].to_numpy(dtype=float), equal_nan=True)
check("Test 2 method flags a leaky feature (shift(-1)) as leaking", leak_changed)

# ----------------------------------------------------------------------
# TEST 4: chronological split
# ----------------------------------------------------------------------
log("\nTEST 4: chronological split")
used = model[model["split"] != "purged"]
blocks = {s: used[used["split"] == s].index for s in ["train", "validation", "test"]}
horizon = pd.Timedelta(hours=1)
check("Rows are in time order", model.index.is_monotonic_increasing)
check("train ends before validation starts, even counting the 1h-ahead answer",
      blocks["train"].max() + horizon < blocks["validation"].min(),
      f"train last t={blocks['train'].max()}  validation first t={blocks['validation'].min()}")
check("validation ends before test starts, even counting the 1h-ahead answer",
      blocks["validation"].max() + horizon < blocks["test"].min(),
      f"validation last t={blocks['validation'].max()}  test first t={blocks['test'].min()}")
total = sum(len(v) for v in blocks.values())
check("No row belongs to two blocks", total == len(used) and len(set().union(*map(set, blocks.values()))) == total)

# ----------------------------------------------------------------------
# TEST 5: no interpolated (partly future-based) values in the modelling data
# ----------------------------------------------------------------------
log("\nTEST 5: interpolated weather values")
filled_any = flags.any(axis=1).reindex(model.index).fillna(False)
check("No modelling row contains an interpolated weather value", int(filled_any.sum()) == 0,
      f"{int(filled_any.sum())} such rows")
check("No missing values in any feature or target", int(model[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum()) == 0)

# ----------------------------------------------------------------------
# TEST 6: target not among inputs
# ----------------------------------------------------------------------
log("\nTEST 6: target is not an input")
check("Target column name is not in FEATURE_COLUMNS", TARGET_COLUMN not in FEATURE_COLUMNS)
identical = [c for c in FEATURE_COLUMNS if np.allclose(model[c], model[TARGET_COLUMN])]
check("No feature is an exact copy of the target", len(identical) == 0, str(identical))

log()
log(f"RESULT: {sum(results)} of {len(results)} checks passed")
REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
raise SystemExit(0 if all(results) else 1)
