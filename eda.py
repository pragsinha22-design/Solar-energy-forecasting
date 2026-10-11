"""
eda.py
======
PHASE 2 - Exploratory Data Analysis.

Run from the project root:
    python src/eda.py

Reads  : data/processed/merged_15min.csv
Writes : results/graphs/01_...png to 08_...png
         results/eda_summary.txt   (the numbers quoted in the report)

All numbers used to describe the graphs are PRINTED by this script, so the
observations in the README / report come from real calculations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "merged_15min.csv"
GRAPH_DIR = PROJECT_ROOT / "results" / "graphs"
SUMMARY_FILE = PROJECT_ROOT / "results" / "eda_summary.txt"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
COLOR = "#1f77b4"
ORANGE = "#ff7f0e"

summary = []


def say(text=""):
    print(text)
    summary.append(str(text))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / name, dpi=150)
    plt.close(fig)
    say(f"  saved {name}")


# ----------------------------------------------------------------------
# Load. We use LOCAL time for plots because "noon" and "morning" are easier
# to read in local clock time than in UTC.
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_FILE, parse_dates=["timestamp", "timestamp_local"])
df = df.set_index("timestamp_local")
df["hour"] = df.index.hour
df["month"] = df.index.month

say(f"Rows: {len(df)}   Local range: {df.index.min()} -> {df.index.max()}")
say("Descriptive statistics:")
say(df[["ac_power", "ambient_temperature", "wind_speed", "wind_direction",
        "poa_irradiance"]].describe().round(2).to_string())

# ----------------------------------------------------------------------
# 1. AC power over time
# ----------------------------------------------------------------------
say("\n[1] AC power over time")
daily_mean = df["ac_power"].resample("D").mean()
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(df.index, df["ac_power"], color=COLOR, linewidth=0.3, alpha=0.5, label="15-min power")
ax.plot(daily_mean.index, daily_mean, color=ORANGE, linewidth=1.5, label="Daily mean")
ax.set_title("AC Power Output Over Time")
ax.set_xlabel("Date (local time)")
ax.set_ylabel("AC power (as recorded in meter file)")
ax.legend(loc="upper right")
save(fig, "01_ac_power_over_time.png")
monthly = df.groupby("month")["ac_power"].mean()
say(f"  Highest-average month: {monthly.idxmax()} ({monthly.max():.1f});"
    f" lowest: {monthly.idxmin()} ({monthly.min():.1f})")
say("  Monthly mean ac_power: " + ", ".join(f"{m}:{v:.0f}" for m, v in monthly.items()))
say(f"  Maximum 15-min value: {df['ac_power'].max():.2f} at {df['ac_power'].idxmax()}")

# ----------------------------------------------------------------------
# 2-4. Power vs weather variables
# ----------------------------------------------------------------------
def scatter(x, xlabel, title, fname, number):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(df[x], df["ac_power"], s=3, alpha=0.15, color=COLOR)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AC power")
    ax.set_title(title)
    save(fig, fname)
    r = df[["ac_power", x]].corr().iloc[0, 1]
    say(f"[{number}] Pearson correlation ac_power vs {x} (all rows): {r:.3f}")


scatter("poa_irradiance", "POA irradiance (W/m²)", "AC Power vs POA Irradiance",
        "02_power_vs_irradiance.png", 2)
scatter("ambient_temperature", "Ambient temperature (as recorded)", "AC Power vs Ambient Temperature",
        "03_power_vs_temperature.png", 3)
scatter("wind_speed", "Wind speed (as recorded)", "AC Power vs Wind Speed",
        "04_power_vs_wind_speed.png", 4)

# Extra numbers to interpret graph 2 and 3 properly
day = df[df["poa_irradiance"] > 50]  # "daytime" = meaningful sunlight
say(f"  Daytime rows (poa_irradiance > 50): {len(day)} of {len(df)}")
say(f"  Daytime correlation power~temperature: {day[['ac_power','ambient_temperature']].corr().iloc[0,1]:.3f}")
say(f"  Daytime correlation power~wind_speed:  {day[['ac_power','wind_speed']].corr().iloc[0,1]:.3f}")
say(f"  Daytime correlation power~irradiance:  {day[['ac_power','poa_irradiance']].corr().iloc[0,1]:.3f}")
# Temperature and sunlight are related (hot afternoons), so show that too
say(f"  Daytime correlation temperature~irradiance: "
    f"{day[['ambient_temperature','poa_irradiance']].corr().iloc[0,1]:.3f}")
# Power per unit of irradiance at high irradiance - does heat reduce it?
high = day[day["poa_irradiance"] > 800].copy()
high["ratio"] = high["ac_power"] / high["poa_irradiance"]
say(f"  Rows with irradiance > 800: {len(high)}; corr(ratio power/irradiance, temperature) = "
    f"{high[['ratio','ambient_temperature']].corr().iloc[0,1]:.3f}")

# ----------------------------------------------------------------------
# 5. Average power by hour of day
# ----------------------------------------------------------------------
say("\n[5] Average power by hour of day (local time)")
by_hour = df.groupby("hour")["ac_power"].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(by_hour.index, by_hour["mean"], color=COLOR)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day (local time)")
ax.set_ylabel("Mean AC power")
ax.set_title("Average Solar Generation by Hour of Day")
save(fig, "05_avg_power_by_hour.png")
say(f"  Peak hour: {by_hour['mean'].idxmax()}:00 with mean {by_hour['mean'].max():.1f}")
producing = by_hour[by_hour["mean"] > 1].index
say(f"  Hours with mean power > 1: {producing.min()}:00 to {producing.max()}:59")

# ----------------------------------------------------------------------
# 6. Correlation heatmap (all rows, and daytime rows only)
# ----------------------------------------------------------------------
say("\n[6] Correlation heatmap")
cols = ["ac_power", "ambient_temperature", "wind_speed", "wind_direction", "poa_irradiance"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title in [(axes[0], df, "All rows"),
                        (axes[1], day, "Daytime rows (irradiance > 50)")]:
    corr = data[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
fig.suptitle("Correlation Between Power and Weather Variables")
save(fig, "06_correlation_heatmap.png")
say("  All rows:\n" + df[cols].corr().round(3).to_string())
say("  Daytime rows:\n" + day[cols].corr().round(3).to_string())

# ----------------------------------------------------------------------
# 7. Distribution of power
# ----------------------------------------------------------------------
say("\n[7] Distribution of solar power")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df["ac_power"], bins=60, color=COLOR, ax=axes[0])
axes[0].set_title("All rows (including night)")
axes[0].set_xlabel("AC power")
sns.histplot(df.loc[df["ac_power"] > 0, "ac_power"], bins=60, color=ORANGE, ax=axes[1])
axes[1].set_title("Only rows with power > 0")
axes[1].set_xlabel("AC power")
save(fig, "07_power_distribution.png")
zero_share = (df["ac_power"] == 0).mean() * 100
say(f"  Share of rows with power exactly 0: {zero_share:.1f}%")
pos = df.loc[df["ac_power"] > 0, "ac_power"]
say(f"  Rows with power > 0: {len(pos)}; mean {pos.mean():.1f}, median {pos.median():.1f}, "
    f"90th percentile {pos.quantile(0.9):.1f}")

# ----------------------------------------------------------------------
# 8. Actual generation over selected days
#    Days are chosen by rule from the data (not hand-picked):
#      - highest daily total, median daily total, lowest 10th-percentile total,
#        and the day with the largest 15-min jumps (most variable).
#    Only complete days (96 rows, no missing irradiance) are considered.
# ----------------------------------------------------------------------
say("\n[8] Selected days")
d = df.copy()
d["date"] = d.index.date
complete = d.groupby("date").filter(lambda g: len(g) == 96 and g["poa_irradiance"].notna().all())
daily_total = complete.groupby("date")["ac_power"].sum()
daily_var = complete.groupby("date")["ac_power"].apply(lambda s: s.diff().abs().sum())
q10 = daily_total.quantile(0.10)
chosen = {
    "Highest daily total": daily_total.idxmax(),
    "Median daily total": (daily_total - daily_total.median()).abs().idxmin(),
    "10th-percentile daily total": (daily_total - q10).abs().idxmin(),
    "Most variable (largest 15-min jumps)": daily_var.idxmax(),
}
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
for ax, (label, date) in zip(axes.ravel(), chosen.items()):
    g = complete[complete["date"] == date]
    ax.plot(g.index, g["ac_power"], color=COLOR, linewidth=1.8)
    ax.fill_between(g.index, g["ac_power"], color=COLOR, alpha=0.15)
    ax.set_title(f"{label}\n{date}")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.set_ylabel("AC power")
    say(f"  {label}: {date}  (daily sum of 15-min values = {daily_total[date]:.0f})")
fig.suptitle("Actual Generation on Selected Days (local time)")
save(fig, "08_selected_days.png")
say(f"  Complete days available: {len(daily_total)}")

SUMMARY_FILE.write_text("\n".join(summary), encoding="utf-8")
say(f"\nSaved summary to {SUMMARY_FILE}")
