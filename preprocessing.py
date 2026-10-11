"""
preprocessing.py
================
PHASE 1 - Load, clean, align and merge the three PVDAQ datasets.

Run from the project root:
    python src/preprocessing.py

Output:
    data/processed/merged_15min.csv        (the merged dataset)
    results/preprocessing_report.txt       (everything printed below, saved)

Every decision is explained in the comments next to the code that does it.
"""

from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
# Settings (change here, not inside the code)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "results" / "preprocessing_report.txt"

METER_FILE = RAW_DIR / "2107_meter_15m_data_2025.csv"
ENV_FILE = RAW_DIR / "2107_environment_15m_data_2025.csv"
IRR_FILE = RAW_DIR / "2107_irradiance_15m_data_2025.csv"

METER_TIMEZONE = "America/Los_Angeles"  # meter file is local wall-clock time (see report)
MAX_FILL_STEPS = 4                       # only fill gaps up to 4 x 15 min = 1 hour

# We collect report lines here so they can be printed AND saved to a file.
report_lines = []


def log(text=""):
    """Print a line and remember it for the report file."""
    print(text)
    report_lines.append(str(text))


def fill_short_gaps(series, max_steps, how="time"):
    """
    Fill missing values ONLY in runs of NaN that are <= max_steps long.
    Longer runs are left completely untouched (still NaN).

    Why not pandas' built-in `limit=`?  `limit` would fill the first few
    values of a long gap and leave the rest, which creates fake partial data.
    Here a long gap stays 100% missing, which is more honest.

    how = "time"  -> straight-line interpolation between the neighbours
    how = "ffill" -> repeat the last known value (used for wind direction,
                     because direction is circular: 359 -> 1 is NOT 180)
    """
    is_nan = series.isna()
    # Give every run of consecutive NaN (or consecutive non-NaN) its own id number
    run_id = (is_nan != is_nan.shift()).cumsum()
    # Length of the run that each row belongs to
    run_length = is_nan.groupby(run_id).transform("sum")
    # Rows that are missing AND belong to a short run
    short_gap = is_nan & (run_length <= max_steps)

    if how == "time":
        filled = series.interpolate(method="time", limit_area="inside")
    else:
        filled = series.ffill()

    result = series.copy()
    result[short_gap] = filled[short_gap]
    return result


def main():
    # ==================================================================
    # STEP 1 - LOAD
    # ==================================================================
    log("=" * 70)
    log("STEP 1: LOAD THE THREE CSV FILES")
    log("=" * 70)
    meter = pd.read_csv(METER_FILE)
    env = pd.read_csv(ENV_FILE)
    irr = pd.read_csv(IRR_FILE)
    log(f"meter      shape: {meter.shape}")
    log(f"environment shape: {env.shape}")
    log(f"irradiance shape: {irr.shape}")

    # ==================================================================
    # STEP 2 - RENAME COLUMNS (readable names)
    # ==================================================================
    meter = meter.rename(columns={
        "measured_on": "timestamp_local",
        "meter_revenue_grade_ac_output_meter_149578": "ac_power",
    })
    env = env.rename(columns={
        "utc_measured_on": "timestamp",
        "ambient_temperature_o_149575": "ambient_temperature",
        "wind_speed_o_149576": "wind_speed",
        "wind_direction_o_149577": "wind_direction",
    })
    irr = irr.rename(columns={
        "utc_measured_on": "timestamp",
        "poa_irradiance_o_149574": "poa_irradiance",
    })

    # ==================================================================
    # STEP 3 - TIMESTAMPS
    # ==================================================================
    log()
    log("=" * 70)
    log("STEP 3: TIMESTAMP ALIGNMENT")
    log("=" * 70)

    # Environment and irradiance are already UTC. Mark them as UTC.
    env["timestamp"] = pd.to_datetime(env["timestamp"]).dt.tz_localize("UTC")
    irr["timestamp"] = pd.to_datetime(irr["timestamp"]).dt.tz_localize("UTC")

    # The meter file is LOCAL wall-clock time (Pacific, with daylight saving).
    # Evidence found during inspection: power peaks at 12:00 while irradiance
    # (UTC) peaks at 20:00, and the meter has 1h15m gaps exactly on the US
    # spring-forward dates.  We therefore:
    #   1) say "these times are America/Los_Angeles local times"
    #   2) convert them to UTC so they match the other two files.
    #
    # ambiguous="NaT": in autumn, 01:00-01:59 happens twice. The file only has
    # ONE copy of that hour, so we cannot know which one it is. We mark those
    # rows as NaT (not-a-time) and drop them rather than guess.
    meter["timestamp_local"] = pd.to_datetime(meter["timestamp_local"])
    meter["timestamp"] = meter["timestamp_local"].dt.tz_localize(
        METER_TIMEZONE, ambiguous="NaT", nonexistent="NaT"
    )
    n_ambiguous = int(meter["timestamp"].isna().sum())
    log(f"Meter rows with an ambiguous/non-existent local time (dropped): {n_ambiguous}")
    if n_ambiguous:
        log("  Their local timestamps:")
        for t in meter.loc[meter["timestamp"].isna(), "timestamp_local"]:
            log(f"    {t}")
    meter = meter.dropna(subset=["timestamp"]).copy()
    meter["timestamp"] = meter["timestamp"].dt.tz_convert("UTC")

    # ==================================================================
    # STEP 4 - IRRADIANCE: make it a clean 15-minute series
    # ==================================================================
    log()
    log("=" * 70)
    log("STEP 4: RESAMPLE IRRADIANCE TO 15 MINUTES")
    log("=" * 70)
    # The raw irradiance has 5-, 10-, 15-, 20-, 30-minute spacing.
    # closed="right", label="right" means the slot labelled 12:15 is the
    # average of readings in (12:00, 12:15].  It only uses readings UP TO the
    # timestamp (never later ones), matching how the meter reports the
    # period ending at its timestamp - so no future information leaks in.
    irr_15 = (
        irr.set_index("timestamp")["poa_irradiance"]
        .resample("15min", closed="right", label="right")
        .mean()
        .to_frame()
    )
    n_empty = int(irr_15["poa_irradiance"].isna().sum())
    log(f"Irradiance rows after resampling: {len(irr_15)}")
    log(f"15-minute slots with no irradiance reading at all: {n_empty}")

    # ==================================================================
    # STEP 5 - MERGE (left join on the meter, which holds the target)
    # ==================================================================
    log()
    log("=" * 70)
    log("STEP 5: MERGE")
    log("=" * 70)
    # Left join: every meter row is kept. Weather is attached when the exact
    # UTC timestamp exists. Weather rows with no matching meter row are not
    # needed (there is no target for them) and are not added.
    df = meter[["timestamp", "timestamp_local", "ac_power"]].merge(
        env, on="timestamp", how="left"
    )
    df = df.merge(irr_15.reset_index(), on="timestamp", how="left")
    df = df.sort_values("timestamp").reset_index(drop=True)

    weather_cols = ["ambient_temperature", "wind_speed", "wind_direction", "poa_irradiance"]

    # Missing values straight after the merge (before any filling)
    before = df[weather_cols + ["ac_power"]].isna().sum()
    log("Missing values right after merge (before filling):")
    for col, n in before.items():
        log(f"  {col:<22}{n}")

    # ==================================================================
    # STEP 6 - MISSING VALUES
    # ==================================================================
    log()
    log("=" * 70)
    log("STEP 6: MISSING VALUES")
    log("=" * 70)
    # Index by time so interpolation uses real time distances.
    df = df.set_index("timestamp")

    # Remember exactly which rows were originally missing, per column.
    originally_missing = {c: df[c].isna() for c in weather_cols}

    # Fill short gaps only (<= 1 hour). Longer gaps are left as NaN.
    df["ambient_temperature"] = fill_short_gaps(df["ambient_temperature"], MAX_FILL_STEPS, "time")
    df["wind_speed"] = fill_short_gaps(df["wind_speed"], MAX_FILL_STEPS, "time")
    df["wind_direction"] = fill_short_gaps(df["wind_direction"], MAX_FILL_STEPS, "ffill")
    df["poa_irradiance"] = fill_short_gaps(df["poa_irradiance"], MAX_FILL_STEPS, "time")

    log(f"Rule: gaps of <= {MAX_FILL_STEPS} consecutive 15-min steps (1 hour) are filled;")
    log("      longer gaps are NOT filled (left as NaN). ac_power is never filled.")
    log()
    log(f"{'column':<22}{'missing before':>16}{'filled':>10}{'still missing':>16}")
    for c in weather_cols:
        b = int(originally_missing[c].sum())
        a = int(df[c].isna().sum())
        log(f"{c:<22}{b:>16}{b - a:>10}{a:>16}")

    # ==================================================================
    # STEP 7 - FINAL CHECKS
    # ==================================================================
    log()
    log("=" * 70)
    log("STEP 7: FINAL CHECKS")
    log("=" * 70)
    df = df.sort_index()
    log(f"Sorted chronologically: {df.index.is_monotonic_increasing}")
    log(f"Duplicate timestamps:   {int(df.index.duplicated().sum())}")
    steps = df.index.to_series().diff().dropna()
    log("Time steps between consecutive rows:")
    for step, count in steps.value_counts().head(6).items():
        log(f"  {str(step):<18}{count}")
    big = steps[steps > pd.Timedelta("15min")]
    log(f"Number of gaps larger than 15 min: {len(big)}")
    for t, s in big.items():
        log(f"  gap of {s} ending at {t}")

    # ==================================================================
    # SAVE + SUMMARY
    # ==================================================================
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = df.reset_index()
    out.to_csv(OUT_DIR / "merged_15min.csv", index=False)

    # Save which weather values were FILLED by us (originally missing, now have
    # a value). Interpolation uses readings from both sides of the gap, i.e.
    # sometimes values from up to 1 hour in the FUTURE. Feature engineering
    # uses this file to exclude those rows, so no future information can leak
    # into the model inputs.
    flags = pd.DataFrame(index=df.index)
    for c in weather_cols:
        flags[c + "_was_filled"] = originally_missing[c] & df[c].notna()
    flags.reset_index().to_csv(OUT_DIR / "filled_flags.csv", index=False)

    log()
    log("=" * 70)
    log("MERGED DATASET SUMMARY")
    log("=" * 70)
    log(f"Shape: {df.shape}")
    log(f"Columns: {list(out.columns)}")
    log(f"Date range (UTC):   {df.index.min()}  ->  {df.index.max()}")
    log(f"Date range (local): {df['timestamp_local'].min()}  ->  {df['timestamp_local'].max()}")
    log()
    log("Remaining missing values per column:")
    for col, n in df.isna().sum().items():
        log(f"  {col:<22}{n}")
    any_nan = df[weather_cols + ["ac_power"]].isna().any(axis=1)
    log(f"Rows with at least one missing value: {int(any_nan.sum())} of {len(df)} "
        f"({any_nan.mean() * 100:.2f}%)")

    # Show exactly where the remaining missing values are
    log()
    log("Where the remaining missing values are (consecutive blocks):")
    for c in weather_cols:
        miss = df[c].isna()
        if not miss.any():
            continue
        block_id = (miss != miss.shift()).cumsum()
        for _, block in df[miss].groupby(block_id[miss]):
            log(f"  {c:<22}{block.index[0]}  ->  {block.index[-1]}   ({len(block)} rows)")

    # Save the report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    log()
    log(f"Saved: {OUT_DIR / 'merged_15min.csv'}")
    log(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
