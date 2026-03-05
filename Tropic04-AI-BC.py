#!/usr/bin/env python3
"""
DHT22 Cold-Storage Monitoring System — Integrated Version
=========================================================
  • DHT22 Temperature & Humidity Sensor (GPIO 4)
  • TROPIC01 Hardware Cryptographic Signing
  • Isolation Forest Anomaly Detection (trained model from train_model.py)
  • IOTA Blockchain Storage
  • Telegram Alerts  ← only fires when a CRITICAL condition is confirmed
      - AI severity == "critical"  (decision score ≤ −0.20)
      - OR  ≥ CONSECUTIVE_ALERT_THRESHOLD consecutive out-of-range readings
  • Daily CSV dispatch via Telegram (every 24 h)
"""

import requests
import json
import time
import logging
import base64
import os
import csv
import subprocess
import hashlib
import numpy as np
import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import deque

import board
import adafruit_dht
import bech32
from nacl.signing import SigningKey
from hashlib import blake2b
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# sklearn is used only to load the pre-trained objects — no re-training at runtime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════════
#  DIRECTORY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR  = Path("/home/pi/DHT22-monitor")
DATA_DIR  = BASE_DIR / "data"
LOG_DIR   = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

for d in (DATA_DIR, LOG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

log_file = LOG_DIR / f"dht22_monitor_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  IOTA BLOCKCHAIN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PACKAGE_ID      = "0xd29d1e0d8a08c14fd76f7345a2526339c5e80728f8dd4117ebc77edfb27d3440"
STORAGE_OBJECT  = "0x879dc5ca2f5fa6610e1d2dbe7cc3382220bdc902274e79e5b7f010e538ddc29c"
ADMIN_CAP_ID    = "0x594cd6544ac8fc5eeeae028c7e2741a3c21587cc78d8f00a590bed28ebb196dd"
CLOCK_OBJECT_ID = "0x0000000000000000000000000000000000000000000000000000000000000006"

PRIVATE_KEY  = "iotaprivkey1qquerfrsgjulthkj7ap3sdnj7lrhufrv9n9t3tq0hm7hckdpkgjv5vyjmun"
GAS_OBJECT_ID = "0x1e9a32dc0f45a65e191d4d431395c31ac1921f8d515217818c01b93f4eca5ff5"
IOTA_RPC_URL  = "https://api.testnet.iota.cafe"
DEVICE_ID     = "DHT22-001"


# ═══════════════════════════════════════════════════════════════════════════════
#  DHT22 SENSOR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DHT_PIN = board.D4   # GPIO 4


# ═══════════════════════════════════════════════════════════════════════════════
#  TROPIC01 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LT_UTIL_PATH   = "/home/pi/libtropic-util/build/lt-util"
TROPIC_KEY_SLOT = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  AES ENCRYPTION
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE_AES_KEY_HEX = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"


# ═══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM CONFIGURATION  (Azmat)
# ═══════════════════════════════════════════════════════════════════════════════

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7588197646:AAHuV-avzr0EnrrTOAsXel8IH7iQ_Gjoito")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "7802792121")
CSV_SEND_INTERVAL  = 86400   # 24 hours


# ═══════════════════════════════════════════════════════════════════════════════
#  COLD-STORAGE TEMPERATURE / HUMIDITY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

TEMP_HIGH_CRITICAL = 6.0    # °C  upper safe limit
TEMP_LOW_CRITICAL  = 1.5    # °C  lower safe limit
HUM_MIN            = 30.0   # %   lower safe limit
HUM_MAX            = 60.0   # %   upper safe limit

# How many consecutive out-of-range readings before a SUSTAINED alert fires
CONSECUTIVE_ALERT_THRESHOLD = 5

# AI decision-score boundary for each severity tier
SCORE_CRITICAL = -0.20
SCORE_HIGH     = -0.12
SCORE_MEDIUM   = -0.07
SCORE_LOW      =  0.00   # anything below this is an anomaly


# ═══════════════════════════════════════════════════════════════════════════════
#  MONITORING TIMING
# ═══════════════════════════════════════════════════════════════════════════════

SENSOR_READ_INTERVAL              = 60      # seconds between sensor reads
CSV_WRITE_INTERVAL                = 300     # seconds between CSV flushes
BLOCKCHAIN_SUBMIT_NORMAL_INTERVAL = 3600    # submit normal reading once per hour
BLOCKCHAIN_SUBMIT_ANOMALY_ALWAYS  = True    # always submit any anomaly to chain


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (must match train_model.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW = 20   # rolling window for statistical features

TEMP_MID = (TEMP_HIGH_CRITICAL + TEMP_LOW_CRITICAL) / 2.0
HUM_MID  = (HUM_MAX + HUM_MIN) / 2.0


def build_feature_vector(temp_history: list, hum_history: list) -> np.ndarray:
    """
    Build the same 9-feature vector used during model training.
    Requires at least 2 readings in history for delta features.
    """
    t = temp_history[-1]
    h = hum_history[-1]

    t_delta = float(t - temp_history[-2]) if len(temp_history) >= 2 else 0.0
    h_delta = float(h - hum_history[-2])  if len(hum_history) >= 2 else 0.0

    heat_idx = t + 0.33 * (h / 100 * 6.105 * np.exp(17.27 * t / (237.7 + t))) - 4.0

    t_from_mid = t - TEMP_MID
    h_from_mid = h - HUM_MID

    window_t = np.array(temp_history[-WINDOW:])
    r_mean   = float(window_t.mean())
    r_std    = float(window_t.std()) if len(window_t) > 1 else 0.0

    return np.array([[t, h, t_delta, h_delta, heat_idx,
                      t_from_mid, h_from_mid, r_mean, r_std]], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
#  DHT22 SENSOR READER
# ═══════════════════════════════════════════════════════════════════════════════

class DHT22Reader:
    def __init__(self):
        self.dht = adafruit_dht.DHT22(DHT_PIN, use_pulseio=False)
        logger.info("DHT22 initialised on GPIO 4")

    def get_reading(self):
        for attempt in range(5):
            try:
                temp     = self.dht.temperature
                humidity = self.dht.humidity
                if temp is not None and humidity is not None:
                    if temp > 80 or humidity > 100:
                        logger.warning(f"Reading out of physical range: {temp}°C {humidity}%")
                        return None
                    return {
                        "temperature":    round(temp, 2),
                        "humidity":       round(humidity, 2),
                        "timestamp":      time.time(),
                        "timestamp_iso":  datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }
            except RuntimeError:
                if attempt < 4:
                    time.sleep(2)
        logger.warning("Failed to read DHT22 after 5 attempts")
        return None

    def cleanup(self):
        self.dht.exit()


# ═══════════════════════════════════════════════════════════════════════════════
#  TROPIC01 SIGNER
# ═══════════════════════════════════════════════════════════════════════════════

class TROPIC01Signer:
    def __init__(self):
        if not os.path.exists(LT_UTIL_PATH):
            raise FileNotFoundError(f"lt-util not found at {LT_UTIL_PATH}")
        self.public_key = self._setup_key()
        logger.info(f"TROPIC01 initialised. Public key prefix: {self.public_key[:32]}…")

    def _setup_key(self):
        pubkey_file = "pubkey.bin"
        result = subprocess.run(
            [LT_UTIL_PATH, "-e", "-d", str(TROPIC_KEY_SLOT), pubkey_file],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.info("Generating new TROPIC01 key…")
            subprocess.run([LT_UTIL_PATH, "-e", "-g", str(TROPIC_KEY_SLOT)], check=True)
            subprocess.run(
                [LT_UTIL_PATH, "-e", "-d", str(TROPIC_KEY_SLOT), pubkey_file], check=True
            )
        with open(pubkey_file, "rb") as f:
            return f.read().hex()

    def sign_data(self, data_dict: dict):
        json_data  = json.dumps(data_dict, sort_keys=True)
        data_hash  = hashlib.sha256(json_data.encode()).digest()
        with open("hash.bin", "wb") as f:
            f.write(data_hash)
        result = subprocess.run(
            [LT_UTIL_PATH, "-e", "-s", str(TROPIC_KEY_SLOT), "hash.bin", "signature.bin"],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.error(f"TROPIC01 signing failed: {result.stderr.decode()}")
            return None, None
        with open("signature.bin", "rb") as f:
            signature = f.read()
        for tmp in ("hash.bin", "signature.bin"):
            if os.path.exists(tmp):
                os.remove(tmp)
        return data_hash, signature


# ═══════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTOR  — uses the pre-trained Isolation Forest + StandardScaler
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Loads the model produced by train_model.py and scores each new reading.
    No retraining happens at runtime — swap the model file to update.
    """

    MODEL_PATH  = MODEL_DIR / "isolation_forest.pkl"
    SCALER_PATH = MODEL_DIR / "scaler.pkl"

    def __init__(self):
        self.model, self.scaler = self._load_model()
        self.temp_history: list  = []
        self.hum_history:  list  = []
        # Track consecutive out-of-range readings for sustained-alert logic
        self.consecutive_violations = 0
        logger.info("AnomalyDetector ready (pre-trained model loaded)")

    def _load_model(self):
        if not self.MODEL_PATH.exists() or not self.SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found in {MODEL_DIR}.\n"
                "Run  python3 train_model.py  first to generate the model files."
            )
        with open(self.MODEL_PATH,  "rb") as f: model  = pickle.load(f)
        with open(self.SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        logger.info(f"Loaded model from {self.MODEL_PATH}")
        return model, scaler

    # ── Rule-based range check ─────────────────────────────────────────────
    def _is_out_of_range(self, temp: float, humidity: float) -> bool:
        return (temp < TEMP_LOW_CRITICAL or temp > TEMP_HIGH_CRITICAL
                or humidity < HUM_MIN    or humidity > HUM_MAX)

    # ── Score-to-severity mapping ──────────────────────────────────────────
    @staticmethod
    def _score_to_severity(score: float) -> str:
        if score <= SCORE_CRITICAL: return "critical"
        if score <= SCORE_HIGH:     return "high"
        if score <= SCORE_MEDIUM:   return "medium"
        if score <= SCORE_LOW:      return "low"
        return "normal"

    # ── Main detection method ──────────────────────────────────────────────
    def detect(self, temperature: float, humidity: float) -> dict | None:
        self.temp_history.append(temperature)
        self.hum_history.append(humidity)
        # Keep a bounded rolling window
        if len(self.temp_history) > WINDOW * 2:
            self.temp_history = self.temp_history[-WINDOW:]
            self.hum_history  = self.hum_history[-WINDOW:]

        # Minimum 2 readings needed for delta features
        if len(self.temp_history) < 2:
            logger.debug("Collecting initial readings (need 2)…")
            return None

        # ── Feature vector ────────────────────────────────────────────────
        X_raw    = build_feature_vector(self.temp_history, self.hum_history)
        X_scaled = self.scaler.transform(X_raw)

        score    = float(self.model.decision_function(X_scaled)[0])
        pred     = int(self.model.predict(X_scaled)[0])     # +1 normal, -1 anomaly
        is_anomaly = (pred == -1)
        severity   = self._score_to_severity(score)

        # ── Consecutive violation counter ─────────────────────────────────
        out_of_range = self._is_out_of_range(temperature, humidity)
        if out_of_range:
            self.consecutive_violations += 1
        else:
            self.consecutive_violations = 0

        # ── Determine which parameter and direction is anomalous ──────────
        parameter  = None
        direction  = None
        anom_value = None
        thresh_val = None

        if is_anomaly:
            temp_mid  = np.mean(self.temp_history[-20:]) if len(self.temp_history) >= 2 else temperature
            hum_mid   = np.mean(self.hum_history[-20:])  if len(self.hum_history) >= 2  else humidity
            temp_dev  = abs(temperature - temp_mid)
            hum_dev   = abs(humidity - hum_mid)

            if temp_dev >= hum_dev:
                parameter  = "temperature"
                anom_value = temperature
                thresh_val = temp_mid
                direction  = "high" if temperature > temp_mid else "low"
            else:
                parameter  = "humidity"
                anom_value = humidity
                thresh_val = hum_mid
                direction  = "high" if humidity > hum_mid else "low"

        # ── Sustained-alert flag ──────────────────────────────────────────
        sustained_alert = self.consecutive_violations >= CONSECUTIVE_ALERT_THRESHOLD

        return {
            "anomaly_score":       score,
            "is_anomaly":          is_anomaly,
            "severity":            severity,
            "parameter":           parameter,
            "anomaly_value":       anom_value,
            "threshold_value":     thresh_val,
            "direction":           direction,
            "out_of_range":        out_of_range,
            "consecutive_violations": self.consecutive_violations,
            "sustained_alert":     sustained_alert,
        }

    def should_send_critical_alert(self, result: dict) -> bool:
        """
        Returns True only when a CRITICAL condition is confirmed:
          1. AI severity == "critical"
          2. OR consecutive out-of-range readings have hit the threshold
        """
        if result is None:
            return False
        return result["severity"] == "critical" or result["sustained_alert"]


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOGGER  (CSV)
# ═══════════════════════════════════════════════════════════════════════════════

class DataLogger:
    def __init__(self):
        today = datetime.now().strftime("%Y-%m-%d")
        self.csv_dir = DATA_DIR
        self.csv_file = DATA_DIR / f"{today}.csv"
        self.buffer: deque = deque(maxlen=200)
        self._ensure_header()

    def _ensure_header(self):
        if not self.csv_file.exists():
            with open(self.csv_file, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "temperature_c", "humidity_percent",
                    "anomaly_score", "is_anomaly", "severity",
                    "consecutive_violations", "sustained_alert",
                    "out_of_range", "tx_digest",
                ])

    def _rotate_if_needed(self):
        today = datetime.now().strftime("%Y-%m-%d")
        expected = DATA_DIR / f"{today}.csv"
        if expected != self.csv_file:
            self.csv_file = expected
            self._ensure_header()

    def add_reading(self, reading: dict, anomaly: dict | None, tx_digest: str | None):
        self.buffer.append({
            **reading,
            "anomaly_score":          anomaly["anomaly_score"]          if anomaly else 0.0,
            "is_anomaly":             anomaly["is_anomaly"]             if anomaly else False,
            "severity":               anomaly["severity"]               if anomaly else "normal",
            "consecutive_violations": anomaly["consecutive_violations"] if anomaly else 0,
            "sustained_alert":        anomaly["sustained_alert"]        if anomaly else False,
            "out_of_range":           anomaly["out_of_range"]           if anomaly else False,
            "tx_digest":              tx_digest or "",
        })

    def flush(self):
        if not self.buffer:
            return
        self._rotate_if_needed()
        try:
            with open(self.csv_file, "a", newline="") as f:
                w = csv.writer(f)
                while self.buffer:
                    e = self.buffer.popleft()
                    w.writerow([
                        datetime.fromtimestamp(e["timestamp"]).isoformat(),
                        e["temperature_c"] if "temperature_c" in e else e.get("temperature"),
                        e["humidity_percent"] if "humidity_percent" in e else e.get("humidity"),
                        round(e["anomaly_score"], 6),
                        e["is_anomaly"],
                        e["severity"],
                        e["consecutive_violations"],
                        e["sustained_alert"],
                        e["out_of_range"],
                        e["tx_digest"],
                    ])
            logger.info("CSV buffer flushed")
        except Exception as ex:
            logger.error(f"CSV flush error: {ex}")

    def calculate_csv_hash(self, path: Path | str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def latest_csv_path(self) -> Path | None:
        """Return path to today's or yesterday's CSV (whichever is non-empty)."""
        today     = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        for name in (today, yesterday):
            p = DATA_DIR / f"{name}.csv"
            if p.exists() and p.stat().st_size > 0:
                return p
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class TelegramManager:
    """
    Sends Telegram messages / documents.
    Alerts are ONLY sent when called from the monitor — this class never
    decides by itself whether to send; it just delivers.
    """

    def __init__(self):
        self.token    = TELEGRAM_BOT_TOKEN
        self.chat_id  = TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.last_csv_send = 0.0

    # ── Alert / Recovery ──────────────────────────────────────────────────

    def send_alert(self, temperature: float, humidity: float,
                   anomaly: dict, severity_reason: str):
        consecutive = anomaly.get("consecutive_violations", 0)
        score       = anomaly.get("anomaly_score", 0)

        if temperature > TEMP_HIGH_CRITICAL:
            breach = "HIGH"
            indicator = "🔴"
        elif temperature < TEMP_LOW_CRITICAL:
            breach = "LOW"
            indicator = "🔵"
        else:
            breach = "HUMIDITY"
            indicator = "🟡"

        message = (
            f"{indicator} *COLD STORAGE CRITICAL ALERT — DEVICE-A*\n\n"
            f"🌡 Temperature : `{temperature:.2f}°C`   (safe: {TEMP_LOW_CRITICAL}–{TEMP_HIGH_CRITICAL}°C)\n"
            f"💧 Humidity    : `{humidity:.1f}%`       (safe: {HUM_MIN}–{HUM_MAX}%)\n"
            f"⚠️ Breach type : {breach}\n"
            f"📊 AI severity : {anomaly['severity'].upper()}\n"
            f"🔁 Consecutive violations : {consecutive}\n"
            f"🧮 AI score    : `{score:.4f}`\n"
            f"📅 Time (UTC)  : `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
            f"_Reason_: {severity_reason}\n\n"
            f"🔴 *IMMEDIATE ACTION REQUIRED!*\n"
            f"Blockchain notification sent ✔"
        )
        return self._send_message(message, parse_mode="Markdown")

    def send_recovery(self, temperature: float, humidity: float,
                      prev_breach: str, breach_duration_s: float):
        hours, rem = divmod(int(breach_duration_s), 3600)
        minutes    = rem // 60
        dur_str    = f"{hours}h {minutes}m" if hours else f"{minutes}m"

        message = (
            f"✅ *TEMPERATURE RETURNED TO NORMAL — DEVICE-A*\n\n"
            f"🌡 Temperature : `{temperature:.2f}°C`\n"
            f"💧 Humidity    : `{humidity:.1f}%`\n"
            f"Previous breach : {prev_breach}\n"
            f"Breach duration : {dur_str}\n"
            f"Time (UTC) : `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
            f"System has returned to normal operation.\n"
            f"Recovery status recorded on blockchain ✔"
        )
        return self._send_message(message, parse_mode="Markdown")

    # ── Daily CSV ─────────────────────────────────────────────────────────

    def maybe_send_daily_csv(self, data_logger: DataLogger,
                              csv_hash_on_chain: str | None = None):
        """Send CSV if 24 h have elapsed since the last send."""
        if time.time() - self.last_csv_send < CSV_SEND_INTERVAL:
            return False
        csv_path = data_logger.latest_csv_path()
        if not csv_path:
            logger.warning("No CSV file ready to send")
            return False

        csv_hash = data_logger.calculate_csv_hash(csv_path)
        caption  = (
            f"📊 *Daily Temperature Report*\n"
            f"Date: `{csv_path.stem}`\n"
            f"Time (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"File hash (SHA-256): `{csv_hash[:16]}…` _(integrity reference)_\n"
            f"Safe range: {TEMP_LOW_CRITICAL}–{TEMP_HIGH_CRITICAL}°C"
        )
        if self._send_document(csv_path, caption):
            self.last_csv_send = time.time()
            logger.info(f"Daily CSV sent: {csv_path}")
            return True
        return False

    # ── Internal helpers ──────────────────────────────────────────────────

    def _send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        try:
            r = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": text,
                      "parse_mode": parse_mode},
                timeout=10,
            )
            if r.status_code == 200:
                logger.info("Telegram message sent ✓")
                return True
            logger.error(f"Telegram API error: {r.text}")
        except Exception as ex:
            logger.error(f"Telegram send error: {ex}")
        return False

    def _send_document(self, path: Path, caption: str) -> bool:
        try:
            with open(path, "rb") as fh:
                r = requests.post(
                    f"{self.base_url}/sendDocument",
                    data={"chat_id": self.chat_id, "caption": caption,
                          "parse_mode": "Markdown"},
                    files={"document": (path.name, fh, "text/csv")},
                    timeout=30,
                )
            return r.status_code == 200
        except Exception as ex:
            logger.error(f"Telegram document send error: {ex}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
#  BLOCKCHAIN MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class BlockchainManager:
    def __init__(self):
        self.last_normal_submit = 0.0
        self.aesgcm = AESGCM(bytes.fromhex(DEVICE_AES_KEY_HEX))

    # ── Key / address helpers ─────────────────────────────────────────────

    def _decode_private_key(self):
        hrp, data = bech32.bech32_decode(PRIVATE_KEY)
        if not hrp or hrp != "iotaprivkey":
            raise ValueError(f"Invalid private key HRP: {hrp}")
        decoded = bech32.convertbits(data, 5, 8, False)
        if not decoded or len(decoded) < 32:
            raise ValueError("Invalid private key length")
        priv = bytes(decoded[-32:])
        return priv, SigningKey(priv).verify_key.encode()

    def get_sender_address(self) -> str:
        _, pub = self._decode_private_key()
        h = blake2b(digest_size=32)
        h.update(pub)
        return "0x" + h.hexdigest()

    def sign_transaction(self, tx_bytes_b64: str) -> str | None:
        try:
            priv, pub = self._decode_private_key()
            tx_data   = base64.b64decode(tx_bytes_b64)
            intent    = b"\x00\x00\x00"
            h = blake2b(digest_size=32)
            h.update(intent + tx_data)
            sig = SigningKey(priv).sign(h.digest()).signature
            serialized = bytes([0x00]) + sig + pub
            return base64.b64encode(serialized).decode()
        except Exception as ex:
            logger.error(f"Transaction signing error: {ex}")
            return None

    def _rpc_call(self, method: str, params: list) -> dict | None:
        try:
            r = requests.post(
                IOTA_RPC_URL,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as ex:
            logger.error(f"RPC {method} error: {ex}")
            return None

    def _execute(self, tx_bytes: str) -> str | None:
        sig = self.sign_transaction(tx_bytes)
        if not sig:
            return None
        result = self._rpc_call(
            "iota_executeTransactionBlock",
            [tx_bytes, [sig],
             {"showInput": True, "showEffects": True,
              "showEvents": True, "showObjectChanges": True}],
        )
        if not result or "error" in result:
            logger.error(f"Execute RPC error: {result}")
            return None

        tx_result = result.get("result", {})
        digest    = tx_result.get("digest")

        # ── Check Move execution status (not just tx acceptance) ──────────
        effects = tx_result.get("effects", {})
        status  = effects.get("status", {})
        exec_status = status.get("status", "unknown")

        if exec_status != "success":
            error_msg = status.get("error", "no error detail returned")
            logger.error(
                f"✗ Tx submitted but contract execution FAILED\n"
                f"  Digest : {digest}\n"
                f"  Reason : {error_msg}\n"
                f"  Hint   : If 'device not found / not authorized', run option 1 "
                f"(Register device) first."
            )
            return None   # return None so the caller knows it failed

        logger.info(f"✓ Tx success: {digest}")
        return digest

    @staticmethod
    def _sv(s: str) -> list:  return list(s.encode("utf-8"))
    @staticmethod
    def _bv(b: bytes) -> list: return list(b)

    # ── Contract calls ────────────────────────────────────────────────────

    def submit_normal_reading(self, temperature: float, humidity: float) -> str | None:
        sender = self.get_sender_address()
        # Contract stores raw integers (TEMP_MAX=80, HUMIDITY_MAX=100)
        temp_f = int(round(temperature))
        hum_f  = int(round(humidity))

        result = self._rpc_call("unsafe_moveCall", [
            sender, PACKAGE_ID, "DHT22Monitor", "store_normal_reading",
            [],
            [STORAGE_OBJECT, self._sv(DEVICE_ID),
             str(temp_f), str(hum_f), CLOCK_OBJECT_ID],
            GAS_OBJECT_ID, "10000000",
        ])
        if not result or "error" in result:
            logger.error(f"Normal reading build error: {result}")
            return None
        digest = self._execute(result["result"]["txBytes"])
        if digest:
            self.last_normal_submit = time.time()
        return digest

    def submit_anomaly_reading(self, temperature: float, humidity: float,
                               anomaly: dict, data_hash: bytes,
                               tropic_sig: bytes) -> str | None:
        sender          = self.get_sender_address()
        # Contract stores raw integers (TEMP_MAX=80, HUMIDITY_MAX=100)
        temp_f          = int(round(temperature))
        hum_f           = int(round(humidity))
        anom_value_f    = int(round(anomaly["anomaly_value"]))
        threshold_f     = int(round(anomaly["threshold_value"]))
        encrypted_value = self._encrypt(anomaly["anomaly_value"])

        result = self._rpc_call("unsafe_moveCall", [
            sender, PACKAGE_ID, "DHT22Monitor", "store_anomaly_reading",
            [],
            [
                STORAGE_OBJECT,
                self._sv(DEVICE_ID),
                str(temp_f), str(hum_f),
                self._sv(anomaly["parameter"]),
                str(anom_value_f),
                str(threshold_f),
                self._sv(anomaly["direction"]),
                self._sv(anomaly["severity"]),
                self._bv(data_hash),
                self._bv(tropic_sig),
                self._bv(encrypted_value),
                CLOCK_OBJECT_ID,
            ],
            GAS_OBJECT_ID, "20000000",
        ])
        if not result or "error" in result:
            logger.error(f"Anomaly build error: {result}")
            return None
        return self._execute(result["result"]["txBytes"])

    def register_device(self, device_id: str) -> str | None:
        sender = self.get_sender_address()
        result = self._rpc_call("unsafe_moveCall", [
            sender, PACKAGE_ID, "DHT22Monitor", "register_device",
            [],
            [ADMIN_CAP_ID, STORAGE_OBJECT,
             self._sv(device_id), sender],
            GAS_OBJECT_ID, "10000000",
        ])
        if not result or "error" in result:
            logger.error(f"Registration build error: {result}")
            return None
        return self._execute(result["result"]["txBytes"])

    def _encrypt(self, value: float) -> bytes:
        nonce = os.urandom(12)
        ct    = self.aesgcm.encrypt(nonce, str(value).encode(), None)
        return nonce + ct


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class DHT22Monitor:
    """
    Orchestrates all subsystems.

    Alert policy (Telegram):
    ─────────────────────────
    A Telegram alert is fired when:
      (a) AI severity == "critical"  (decision score ≤ −0.20)
      (b) OR ≥ CONSECUTIVE_ALERT_THRESHOLD consecutive out-of-range readings

    A recovery message is sent once the condition clears.
    Daily CSV is sent once every 24 h regardless of anomaly state.
    """

    def __init__(self):
        self.sensor      = DHT22Reader()
        self.signer      = TROPIC01Signer()
        self.detector    = AnomalyDetector()
        self.data_logger = DataLogger()
        self.blockchain  = BlockchainManager()
        self.telegram    = TelegramManager()

        self.running          = False
        self.last_csv_flush   = time.time()
        self.reading_counter  = 0

        # ── Anomaly state machine ─────────────────────────────────────────
        # Blockchain + Telegram fire ONCE on entry and ONCE on recovery.
        # Nothing is sent for intermediate readings while anomaly persists.
        self._in_anomaly      = False      # True while a critical condition is active
        self._anomaly_start   = 0.0        # time.time() when anomaly began
        self._anomaly_breach  = "UNKNOWN"  # "HIGH" | "LOW" | "HUMIDITY"

        # ── Warn immediately if device looks unregistered ─────────────────
        self._check_device_registered()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _check_device_registered(self):
        """
        Query the on-chain storage object and check whether DEVICE_ID appears
        in it.  This is a read-only getDynamicFields call — it spends no gas.
        If the device is not found we print a clear warning before the loop
        starts, so the user knows to run option 1 first.
        """
        try:
            result = self.blockchain._rpc_call(
                "iota_getDynamicFields",
                [STORAGE_OBJECT, None, 10],
            )
            if not result or "error" in result:
                logger.warning(
                    "Could not query on-chain storage to verify device registration. "
                    "Continuing anyway."
                )
                return
            fields = result.get("result", {}).get("data", [])
            device_names = [
                f.get("name", {}).get("value", "") for f in fields
            ]
            if DEVICE_ID not in device_names:
                logger.warning(
                    f"\n{'='*60}\n"
                    f"  DEVICE '{DEVICE_ID}' IS NOT REGISTERED ON-CHAIN!\n"
                    f"  All store_normal_reading / store_anomaly_reading calls\n"
                    f"  will FAIL until you run option 1 (Register device).\n"
                    f"  Stop the monitor (Ctrl-C), run it again, choose option 1,\n"
                    f"  then restart monitoring.\n"
                    f"{'='*60}"
                )
            else:
                logger.info(f"✓ Device '{DEVICE_ID}' confirmed registered on-chain.")
        except Exception as ex:
            logger.warning(f"Device registration check error (non-fatal): {ex}")


    def _should_submit_normal(self) -> bool:
        return (time.time() - self.blockchain.last_normal_submit) >= BLOCKCHAIN_SUBMIT_NORMAL_INTERVAL

    def _alert_reason(self, anomaly: dict) -> str:
        reasons = []
        if anomaly["severity"] == "critical":
            reasons.append(f"AI score {anomaly['anomaly_score']:.4f} ≤ {SCORE_CRITICAL}")
        if anomaly["sustained_alert"]:
            reasons.append(
                f"{anomaly['consecutive_violations']} consecutive out-of-range readings "
                f"(threshold: {CONSECUTIVE_ALERT_THRESHOLD})"
            )
        return " | ".join(reasons) if reasons else "critical condition detected"

    def _determine_breach_type(self, temperature: float, humidity: float) -> str:
        if temperature > TEMP_HIGH_CRITICAL:   return "HIGH"
        if temperature < TEMP_LOW_CRITICAL:    return "LOW"
        return "HUMIDITY"

    # ── Sign data helper ──────────────────────────────────────────────────

    def _sign_reading(self, reading: dict):
        payload = {
            "temperature_c":    reading["temperature"],
            "humidity_percent": reading["humidity"],
            "timestamp":        reading["timestamp_iso"],
        }
        return self.signer.sign_data(payload)

    # ── Main loop ─────────────────────────────────────────────────────────

    def monitor_loop(self):
        logger.info("=" * 70)
        logger.info("DHT22 Cold-Storage Monitor starting")
        logger.info(f"Safe range: {TEMP_LOW_CRITICAL}\u2013{TEMP_HIGH_CRITICAL}\u00b0C  |  "
                    f"{HUM_MIN}\u2013{HUM_MAX}%")
        logger.info(f"Alert triggers: AI severity=critical  OR  "
                    f"\u2265{CONSECUTIVE_ALERT_THRESHOLD} consecutive violations")
        logger.info(f"Blockchain: anomaly submitted ONCE on entry, "
                    f"normal submitted ONCE on recovery & every {BLOCKCHAIN_SUBMIT_NORMAL_INTERVAL}s")
        logger.info("=" * 70)
        self.running = True

        while self.running:
            try:
                # ── 1. Read sensor ────────────────────────────────────────
                reading = self.sensor.get_reading()
                if not reading:
                    time.sleep(SENSOR_READ_INTERVAL)
                    continue

                temp = reading["temperature"]
                hum  = reading["humidity"]
                logger.info(
                    f"Reading #{self.reading_counter + 1} \u2014 "
                    f"Temp: {temp}\u00b0C  Humidity: {hum}%"
                    + (f"  [ANOMALY ACTIVE]" if self._in_anomaly else "")
                )

                # ── 2. Anomaly detection ──────────────────────────────────
                anomaly = self.detector.detect(temp, hum)

                if anomaly:
                    logger.info(
                        f"  AI score={anomaly['anomaly_score']:.4f}  "
                        f"severity={anomaly['severity']}  "
                        f"consecutive={anomaly['consecutive_violations']}  "
                        f"sustained={anomaly['sustained_alert']}"
                    )

                # ── 3. Cryptographic signing (always needed for state transitions)
                data_hash, tropic_sig = self._sign_reading(reading)
                if data_hash is None:
                    time.sleep(SENSOR_READ_INTERVAL)
                    continue

                tx_digest   = None
                is_critical = (anomaly is not None and
                               self.detector.should_send_critical_alert(anomaly))

                # ════════════════════════════════════════════════════════
                #  STATE MACHINE
                #
                #  NORMAL → ANOMALY  : submit anomaly tx once + Telegram alert once
                #  ANOMALY (ongoing) : log only, no blockchain, no Telegram
                #  ANOMALY → NORMAL  : submit normal tx once + Telegram recovery once
                #  NORMAL (ongoing)  : submit normal tx every BLOCKCHAIN_SUBMIT_NORMAL_INTERVAL
                # ════════════════════════════════════════════════════════

                if is_critical and not self._in_anomaly:
                    # ── Transition: NORMAL → ANOMALY ─────────────────────
                    self._in_anomaly     = True
                    self._anomaly_start  = time.time()
                    self._anomaly_breach = self._determine_breach_type(temp, hum)

                    # Blockchain: one anomaly record
                    tx_digest = self.blockchain.submit_anomaly_reading(
                        temp, hum, anomaly, data_hash, tropic_sig
                    )
                    if tx_digest:
                        logger.info(f"  \u2192 Anomaly ENTRY recorded on-chain: {tx_digest}")
                    else:
                        logger.error("  Anomaly blockchain submission failed")

                    # Telegram: one alert
                    reason = self._alert_reason(anomaly)
                    logger.warning(f"CRITICAL ALERT triggered: {reason}")
                    self.telegram.send_alert(temp, hum, anomaly, reason)

                elif is_critical and self._in_anomaly:
                    # ── Anomaly ongoing — suppress blockchain + Telegram ──
                    elapsed = int(time.time() - self._anomaly_start)
                    logger.info(
                        f"  Anomaly ongoing ({elapsed}s) — "
                        f"blockchain/Telegram suppressed until recovery"
                    )

                elif not is_critical and self._in_anomaly:
                    # ── Transition: ANOMALY → NORMAL ─────────────────────
                    duration             = time.time() - self._anomaly_start
                    prev_breach          = self._anomaly_breach
                    self._in_anomaly     = False
                    self._anomaly_start  = 0.0
                    self._anomaly_breach = "UNKNOWN"

                    # Blockchain: one normal recovery record
                    tx_digest = self.blockchain.submit_normal_reading(temp, hum)
                    if tx_digest:
                        logger.info(f"  \u2192 Recovery recorded on-chain: {tx_digest}")
                    else:
                        logger.error("  Recovery blockchain submission failed")

                    # Telegram: one recovery message
                    logger.info(f"Recovery after {duration:.0f}s — sending notification")
                    self.telegram.send_recovery(temp, hum, prev_breach, duration)

                else:
                    # ── Normal ongoing — periodic heartbeat submission ────
                    if self._should_submit_normal():
                        tx_digest = self.blockchain.submit_normal_reading(temp, hum)
                        if tx_digest:
                            logger.info(f"  \u2192 Periodic normal reading on-chain: {tx_digest}")

                # ── 6. Daily CSV dispatch ─────────────────────────────────
                self.telegram.maybe_send_daily_csv(self.data_logger)

                # ── 7. Log to CSV ─────────────────────────────────────────
                self.data_logger.add_reading(reading, anomaly, tx_digest)

                if (time.time() - self.last_csv_flush) >= CSV_WRITE_INTERVAL:
                    self.data_logger.flush()
                    self.last_csv_flush = time.time()

                self.reading_counter += 1
                time.sleep(SENSOR_READ_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Stopped by user (Ctrl-C)")
                break
            except Exception as ex:
                logger.error(f"Monitor loop error: {ex}", exc_info=True)
                time.sleep(SENSOR_READ_INTERVAL)

        self.running = False

    def stop(self):
        logger.info("Shutting down DHT22 monitor…")
        self.running = False
        self.data_logger.flush()
        self.sensor.cleanup()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("IOTA DHT22 Cold-Storage Monitor — Integrated (Telegram + AI)")
    print("=" * 70)
    print(f"Package ID     : {PACKAGE_ID}")
    print(f"Storage Object : {STORAGE_OBJECT}")
    print(f"Device ID      : {DEVICE_ID}")
    print(f"Safe range     : {TEMP_LOW_CRITICAL}–{TEMP_HIGH_CRITICAL}°C  |  {HUM_MIN}–{HUM_MAX}%")
    print(f"Alert triggers : AI critical score ≤ {SCORE_CRITICAL}  "
          f"OR  ≥{CONSECUTIVE_ALERT_THRESHOLD} consecutive violations")
    print("=" * 70)
    print()
    print("1. Register device (requires AdminCap)")
    print("2. Start monitoring")
    print("3. Test DHT22 sensor")
    print("4. Test TROPIC01 signing")
    print("5. Test Telegram message")

    choice = input("\nSelect option> ").strip()

    if choice == "1":
        blockchain = BlockchainManager()
        digest = blockchain.register_device(DEVICE_ID)
        print(f"\n{'✓ Device registered! Tx: ' + digest if digest else '✗ Registration failed'}")

    elif choice == "2":
        print("\nInitialising monitoring system…")
        monitor = DHT22Monitor()
        print(f"\nSensor interval  : {SENSOR_READ_INTERVAL}s")
        print(f"Normal tx every  : {BLOCKCHAIN_SUBMIT_NORMAL_INTERVAL}s")
        print(f"Anomaly tx       : immediate")
        print(f"Daily CSV send   : every 24 h")
        print("\nPress Ctrl+C to stop\n")
        try:
            monitor.monitor_loop()
        except KeyboardInterrupt:
            monitor.stop()
            print("\nMonitoring stopped.")

    elif choice == "3":
        reader  = DHT22Reader()
        reading = reader.get_reading()
        if reading:
            print(f"\n✓ DHT22 — Temp: {reading['temperature']}°C  Humidity: {reading['humidity']}%")
        else:
            print("\n✗ Failed to read DHT22")
        reader.cleanup()

    elif choice == "4":
        signer = TROPIC01Signer()
        test   = {"temperature_c": 3.5, "humidity_percent": 45.0,
                  "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
        h, s   = signer.sign_data(test)
        if h and s:
            print(f"\n✓ Hash:      {h.hex()[:64]}…")
            print(f"  Signature: {s.hex()[:64]}…")
        else:
            print("\n✗ Signing failed")

    elif choice == "5":
        tg = TelegramManager()
        ok = tg._send_message("✅ DHT22 Monitor test message — connection OK")
        print(f"\n{'✓ Telegram OK' if ok else '✗ Telegram failed'}")


if __name__ == "__main__":
    main()