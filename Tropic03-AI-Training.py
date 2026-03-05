#!/usr/bin/env python3

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("[ERROR] scikit-learn not installed.")
    print("        Run: pip install scikit-learn pandas numpy")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← edit if your environment differs
# ═══════════════════════════════════════════════════════════════════════════════

# Where Tropic01.py writes the lightweight CSVs
TRAINING_DIR = Path("/home/pi/DHT22-monitor/training")

# Where to save the trained model files
MODEL_DIR = Path("/home/pi/DHT22-monitor/models")

# ── Safe operating range for the cold storage unit ───────────────────────────
TEMP_MIN  = 1.6    # °C  — lower safe limit
TEMP_MAX  = 6.0    # °C  — upper safe limit
HUM_MIN   = 30.0   # %RH — lower safe limit
HUM_MAX   = 60.0   # %RH — upper safe limit

# ── Continuous violation alert settings ───────────────────────────────────────
# How many consecutive out-of-range readings before raising a sustained alert
CONSECUTIVE_ALERT_THRESHOLD = 5   # e.g. 5 readings in a row = alert

# ── Isolation Forest hyperparameters ─────────────────────────────────────────
CONTAMINATION = 0.05   # expected fraction of anomalies in training data
N_ESTIMATORS  = 100    # number of trees
RANDOM_STATE  = 42

# Rolling window size for statistical features
WINDOW = 20

# ═══════════════════════════════════════════════════════════════════════════════


MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH  = MODEL_DIR / "isolation_forest.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

FEATURE_NAMES = [
    "temperature_c",
    "humidity_percent",
    "temp_delta",          # change from previous reading
    "humidity_delta",      # change from previous reading
    "heat_index",          # thermal-comfort proxy
    "temp_from_mid",       # distance from centre of safe temp range
    "hum_from_mid",        # distance from centre of safe humidity range
    "rolling_temp_mean",   # rolling mean over last WINDOW readings
    "rolling_temp_std",    # rolling std  over last WINDOW readings
]


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Load one specific CSV  or  all *.csv files from TRAINING_DIR.
    Returns a sorted, deduplicated DataFrame.
    """
    if csv_path:
        files = [csv_path]
    else:
        pattern = str(TRAINING_DIR / "*.csv")   # matches date-named files
        files   = sorted(glob.glob(pattern))

    if not files:
        print(f"\n[ERROR] No CSV files found.")
        if not csv_path:
            print(f"        Expected location: {TRAINING_DIR}/*.csv")
            print(f"        Run Tropic01.py first to collect data.")
        sys.exit(1)

    print(f"\n[LOAD]  Found {len(files)} CSV file(s):")
    frames = []
    for f in files:
        df_part = pd.read_csv(f)
        print(f"        {Path(f).name}  →  {len(df_part):,} rows")
        frames.append(df_part)

    df = pd.concat(frames, ignore_index=True)
    print(f"        ─────────────────────────────")
    print(f"        Total rows (combined) : {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Identify temperature and humidity columns (flexible naming)
    - Convert to numeric, drop NaN rows
    - Sort by timestamp if present
    - Remove duplicate rows
    """
    temp_candidates = ["temperature_c", "temperature", "temp", "Temperature",
                       "temp_c", "TEMP"]
    hum_candidates  = ["humidity_percent", "humidity", "hum", "rh",
                       "Humidity", "humidity_pct", "RH"]

    def find_col(candidates, label):
        col_lower = {c.lower(): c for c in df.columns}
        for name in candidates:
            if name in df.columns:
                return name
            if name.lower() in col_lower:
                return col_lower[name.lower()]
        print(f"[ERROR] Cannot find {label} column in CSV.")
        print(f"        Available columns: {list(df.columns)}")
        sys.exit(1)

    temp_col = find_col(temp_candidates, "temperature")
    hum_col  = find_col(hum_candidates,  "humidity")

    print(f"\n[CLEAN] Using columns: temperature='{temp_col}'  humidity='{hum_col}'")

    df = df.rename(columns={temp_col: "temperature_c", hum_col: "humidity_percent"})

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    before = len(df)
    df["temperature_c"]    = pd.to_numeric(df["temperature_c"],    errors="coerce")
    df["humidity_percent"] = pd.to_numeric(df["humidity_percent"], errors="coerce")
    df = df.dropna(subset=["temperature_c", "humidity_percent"]).reset_index(drop=True)

    df = df.drop_duplicates(
        subset=["temperature_c", "humidity_percent"] +
               (["timestamp"] if "timestamp" in df.columns else [])
    ).reset_index(drop=True)

    dropped = before - len(df)
    if dropped:
        print(f"[CLEAN] Dropped {dropped} invalid/duplicate rows.")

    t_min, t_max = df["temperature_c"].min(), df["temperature_c"].max()
    h_min, h_max = df["humidity_percent"].min(), df["humidity_percent"].max()
    print(f"[CLEAN] Clean rows       : {len(df):,}")
    print(f"[CLEAN] Temperature range: {t_min:.1f} – {t_max:.1f} °C")
    print(f"[CLEAN] Humidity range   : {h_min:.1f} – {h_max:.1f} %")

    if t_max > 85:
        print("[WARN]  Very high temperatures detected — check units (should be °C).")
    if h_max > 100 or h_min < 0:
        print("[WARN]  Humidity out of 0–100 range — check CSV units.")

    if len(df) < 50:
        print(f"\n[ERROR] Only {len(df)} clean rows — need at least 50 to train.")
        sys.exit(1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Expand each row into a 9-element feature vector.
    Computed in time order so rolling statistics are meaningful.
    """
    temps = df["temperature_c"].values.astype(float)
    hums  = df["humidity_percent"].values.astype(float)
    n     = len(temps)

    temp_mid = (TEMP_MAX + TEMP_MIN) / 2.0
    hum_mid  = (HUM_MAX  + HUM_MIN)  / 2.0

    rows = []
    for i in range(n):
        t = temps[i]
        h = hums[i]

        t_delta  = float(t - temps[i - 1]) if i > 0 else 0.0
        h_delta  = float(h - hums[i - 1])  if i > 0 else 0.0

        heat_idx = t + 0.33 * (h / 100 * 6.105 * np.exp(17.27 * t / (237.7 + t))) - 4.0

        t_from_mid = t - temp_mid
        h_from_mid = h - hum_mid

        start  = max(0, i - WINDOW + 1)
        window = temps[start : i + 1]
        r_mean = float(window.mean())
        r_std  = float(window.std()) if len(window) > 1 else 0.0

        rows.append([t, h, t_delta, h_delta, heat_idx,
                     t_from_mid, h_from_mid, r_mean, r_std])

    return np.array(rows, dtype=float)


def rule_violation_mask(df: pd.DataFrame) -> np.ndarray:
    """
    Returns 1 for every reading that violates the safe temperature/humidity
    bounds. Used in the evaluation report only — NOT used to train the model.
    """
    t = df["temperature_c"].values
    h = df["humidity_percent"].values
    return ((t < TEMP_MIN) | (t > TEMP_MAX) |
            (h < HUM_MIN)  | (h > HUM_MAX)).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
#  CONTINUOUS VIOLATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_continuous_violations(df: pd.DataFrame,
                                  threshold: int = CONSECUTIVE_ALERT_THRESHOLD):
    """
    Scan the data in time order and identify runs of consecutive out-of-range
    readings.  A run of >= `threshold` consecutive violations is flagged as a
    SUSTAINED ALERT event.

    Returns a list of alert dicts, each with:
        start_idx, end_idx, length,
        direction ('TOO HIGH' / 'TOO LOW' / 'HUMIDITY'),
        temp_min, temp_max, hum_min, hum_max  (for that window)
    and a per-row integer array: 0=ok, 1=single violation, 2=sustained alert
    """
    t = df["temperature_c"].values
    h = df["humidity_percent"].values
    n = len(t)

    # Per-row violation flag
    temp_too_high = t > TEMP_MAX
    temp_too_low  = t < TEMP_MIN
    hum_out       = (h < HUM_MIN) | (h > HUM_MAX)
    any_violation = temp_too_high | temp_too_low | hum_out

    severity = np.zeros(n, dtype=int)   # 0=ok, 1=violation, 2=sustained
    severity[any_violation] = 1

    alerts = []
    i = 0
    while i < n:
        if any_violation[i]:
            # find end of this run
            j = i
            while j < n and any_violation[j]:
                j += 1
            run_len = j - i
            if run_len >= threshold:
                # mark as sustained
                severity[i:j] = 2
                window_t = t[i:j]
                window_h = h[i:j]

                # determine primary direction
                if temp_too_high[i:j].sum() > temp_too_low[i:j].sum():
                    direction = "TOO HIGH"
                elif temp_too_low[i:j].sum() > 0:
                    direction = "TOO LOW"
                else:
                    direction = "HUMIDITY"

                ts_start = df["timestamp"].iloc[i] if "timestamp" in df.columns else None
                ts_end   = df["timestamp"].iloc[j-1] if "timestamp" in df.columns else None

                alerts.append({
                    "start_idx"  : i,
                    "end_idx"    : j - 1,
                    "length"     : run_len,
                    "direction"  : direction,
                    "temp_min"   : float(window_t.min()),
                    "temp_max"   : float(window_t.max()),
                    "hum_min"    : float(window_h.min()),
                    "hum_max"    : float(window_h.max()),
                    "ts_start"   : ts_start,
                    "ts_end"     : ts_end,
                })
            i = j
        else:
            i += 1

    return alerts, severity


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(X: np.ndarray, contamination: float):
    """Fit StandardScaler + IsolationForest. Return (model, scaler, X_scaled)."""
    print(f"\n[TRAIN] Fitting StandardScaler on {X.shape[0]:,} samples × "
          f"{X.shape[1]} features ...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[TRAIN] Fitting IsolationForest  "
          f"(n_estimators={N_ESTIMATORS}, contamination={contamination}) ...")
    model = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        contamination = contamination,
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    model.fit(X_scaled)
    print("[TRAIN] Done.")
    return model, scaler, X_scaled


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, scaler):
    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"\n[SAVED] {MODEL_PATH}")
    print(f"[SAVED] {SCALER_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(model, scaler, X_scaled: np.ndarray,
                 rule_flags: np.ndarray, df: pd.DataFrame,
                 alerts: list, severity: np.ndarray):

    preds  = model.predict(X_scaled)          # +1 normal, -1 anomaly
    scores = model.decision_function(X_scaled)
    ai_anom  = preds == -1
    n        = len(X_scaled)
    n_ai     = ai_anom.sum()
    n_rule   = rule_flags.sum()

    print("\n" + "═" * 64)
    print("  TRAINING REPORT")
    print("═" * 64)
    print(f"  Samples trained on            : {n:,}")
    print(f"  Safe temperature range        : {TEMP_MIN} – {TEMP_MAX} °C")
    print(f"  Safe humidity range           : {HUM_MIN} – {HUM_MAX} %")
    print(f"  Temperature range in data     : "
          f"{df['temperature_c'].min():.1f} – {df['temperature_c'].max():.1f} °C")
    print(f"  Humidity range in data        : "
          f"{df['humidity_percent'].min():.1f} – {df['humidity_percent'].max():.1f} %")
    print()
    print(f"  Rule-based violations         : {n_rule:,}  ({100*n_rule/n:.1f}%)")
    print(f"  AI anomalies flagged          : {n_ai:,}  ({100*n_ai/n:.1f}%)")
    print()
    print(f"  Decision score  min           : {scores.min():.4f}")
    print(f"  Decision score  max           : {scores.max():.4f}")
    print(f"  Decision score  mean          : {scores.mean():.4f}")
    print(f"  Decision score  std           : {scores.std():.4f}")

    if n_rule > 0:
        both      = (ai_anom & rule_flags.astype(bool)).sum()
        only_ai   = (ai_anom & ~rule_flags.astype(bool)).sum()
        only_rule = (~ai_anom & rule_flags.astype(bool)).sum()
        print()
        print(f"  AI caught {both}/{n_rule} rule violations "
              f"({100*both/n_rule:.0f}%)")
        print(f"  Pattern-only (AI, not rule)   : {only_ai}  "
              f"← anomalies thresholds alone would miss")
        print(f"  Missed by AI (rule-only)      : {only_rule}  "
              f"← consider lowering contamination")

    # ── Continuous / sustained violation alerts ───────────────────────────
    print()
    print("─" * 64)
    print(f"  CONTINUOUS VIOLATION ALERTS  "
          f"(≥ {CONSECUTIVE_ALERT_THRESHOLD} consecutive out-of-range readings)")
    print("─" * 64)

    n_sustained = (severity == 2).sum()

    if not alerts:
        print(f"  ✓  No sustained violation events found in training data.")
        print(f"     (Single/isolated violations: {(severity==1).sum():,} readings)")
    else:
        print(f"  ⚠  {len(alerts)} sustained alert event(s) detected  "
              f"({n_sustained:,} readings affected)\n")
        for idx, alert in enumerate(alerts, 1):
            direction = alert["direction"]
            length    = alert["length"]
            t_lo      = alert["temp_min"]
            t_hi      = alert["temp_max"]
            h_lo      = alert["hum_min"]
            h_hi      = alert["hum_max"]

            # severity badge
            if direction == "TOO HIGH" and t_hi > TEMP_MAX + 4:
                badge = "🔴 CRITICAL"
            elif direction == "TOO HIGH":
                badge = "🟠 HIGH"
            elif direction == "TOO LOW":
                badge = "🔵 LOW"
            else:
                badge = "🟡 HUMIDITY"

            print(f"  Alert #{idx}  {badge}  —  {length} consecutive readings")
            print(f"    Temperature : {t_lo:.1f} – {t_hi:.1f} °C  "
                  f"(safe: {TEMP_MIN}–{TEMP_MAX} °C)  →  TEMPERATURE {direction}")
            print(f"    Humidity    : {h_lo:.1f} – {h_hi:.1f} %  "
                  f"(safe: {HUM_MIN}–{HUM_MAX} %)")
            if alert["ts_start"] is not None:
                print(f"    From        : {alert['ts_start']}")
                print(f"    To          : {alert['ts_end']}")
            print()

        # Summary of direction breakdown
        too_high = sum(1 for a in alerts if a["direction"] == "TOO HIGH")
        too_low  = sum(1 for a in alerts if a["direction"] == "TOO LOW")
        hum_only = sum(1 for a in alerts if a["direction"] == "HUMIDITY")
        print(f"  Direction breakdown:")
        print(f"    Too High (> {TEMP_MAX}°C)  : {too_high} event(s)")
        print(f"    Too Low  (< {TEMP_MIN}°C)  : {too_low} event(s)")
        print(f"    Humidity only             : {hum_only} event(s)")
        print()
        if too_high > 0:
            print(f"  ⚠  WARNING: Fridge temperatures exceeded {TEMP_MAX}°C during training data.")
            print(f"     The model has learned from these warm periods.")
            print(f"     Consider retraining on only known-good cold periods for a")
            print(f"     cleaner baseline.")

    # ── Feature importance ────────────────────────────────────────────────
    print()
    print("─" * 64)
    print("  Feature signal (mean |scaled value|) — higher = more variable:")
    importance = np.abs(X_scaled).mean(axis=0)
    for name, score in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
        bar = "█" * min(30, int(score * 15))
        print(f"    {name:<22}  {bar:<30}  {score:.3f}")

    # ── Severity thresholds ───────────────────────────────────────────────
    print()
    print("─" * 64)
    print("  AI anomaly severity thresholds (decision score):")
    print("    CRITICAL  score ≤ −0.20  ← fridge likely failing / door left open")
    print("    HIGH      score ≤ −0.12")
    print("    MEDIUM    score ≤ −0.07")
    print("    LOW       score ≤  0.00  ← anomaly threshold")
    print()
    print("  Rule-based alert triggers:")
    print(f"    TEMP ALERT  temperature < {TEMP_MIN}°C  or  > {TEMP_MAX}°C")
    print(f"    HUM  ALERT  humidity    < {HUM_MIN}%   or  > {HUM_MAX}%")
    print(f"    SUSTAINED   {CONSECUTIVE_ALERT_THRESHOLD}+ consecutive readings outside safe range")

    print()
    print("═" * 64)
    print("  ✓  Model saved.  Ready for monitoring.")
    print()
    print("  NEXT STEP — run the monitor:")
    print(f"    python3 dht22_monitor.py --continuous 300")
    print("═" * 64)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global TEMP_MIN, TEMP_MAX, HUM_MIN, HUM_MAX, CONSECUTIVE_ALERT_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Train Isolation Forest model from DHT22 cold-storage CSV data."
    )
    parser.add_argument(
        "--csv", default=None,
        help=("Path to a single CSV file. "
              "If omitted, all *.csv files in "
              f"{TRAINING_DIR} are loaded and combined.")
    )
    parser.add_argument(
        "--contamination", type=float, default=CONTAMINATION,
        help=f"Expected fraction of anomalies in data (default: {CONTAMINATION})"
    )
    parser.add_argument(
        "--temp-min", type=float, default=TEMP_MIN,
        help=f"Lower safe temperature bound °C (default: {TEMP_MIN})"
    )
    parser.add_argument(
        "--temp-max", type=float, default=TEMP_MAX,
        help=f"Upper safe temperature bound °C (default: {TEMP_MAX})"
    )
    parser.add_argument(
        "--hum-min", type=float, default=HUM_MIN,
        help=f"Lower safe humidity bound %% (default: {HUM_MIN})"
    )
    parser.add_argument(
        "--hum-max", type=float, default=HUM_MAX,
        help=f"Upper safe humidity bound %% (default: {HUM_MAX})"
    )
    parser.add_argument(
        "--consecutive", type=int, default=CONSECUTIVE_ALERT_THRESHOLD,
        help=f"Consecutive out-of-range readings to trigger sustained alert "
             f"(default: {CONSECUTIVE_ALERT_THRESHOLD})"
    )
    args = parser.parse_args()

    # Apply CLI overrides
    TEMP_MIN  = args.temp_min
    TEMP_MAX  = args.temp_max
    HUM_MIN   = args.hum_min
    HUM_MAX   = args.hum_max
    CONSECUTIVE_ALERT_THRESHOLD = args.consecutive

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   DHT22 Isolation Forest — Model Training                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Source        : {args.csv if args.csv else TRAINING_DIR / '*.csv'}")
    print(f"  Safe temp     : {TEMP_MIN} – {TEMP_MAX} °C")
    print(f"  Safe humidity : {HUM_MIN}  – {HUM_MAX}  %")
    print(f"  Contamination : {args.contamination}")
    print(f"  Sustained alert after : {CONSECUTIVE_ALERT_THRESHOLD} consecutive violations")

    # 1. Load
    df = load_data(args.csv)

    # 2. Clean
    df = clean_data(df)

    # 3. Features
    print(f"\n[FEAT]  Engineering {len(FEATURE_NAMES)} features ...")
    X          = build_features(df)
    rule_flags = rule_violation_mask(df)
    print(f"[FEAT]  Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")

    # 4. Continuous violation analysis
    print(f"\n[ALERT] Scanning for continuous violations "
          f"(threshold: {CONSECUTIVE_ALERT_THRESHOLD} consecutive readings) ...")
    alerts, severity = analyse_continuous_violations(df, CONSECUTIVE_ALERT_THRESHOLD)
    print(f"[ALERT] Sustained alert events found: {len(alerts)}")

    # 5. Train
    model, scaler, X_scaled = train(X, args.contamination)

    # 6. Save
    save_model(model, scaler)

    # 7. Report
    print_report(model, scaler, X_scaled, rule_flags, df, alerts, severity)


if __name__ == "__main__":
    main()