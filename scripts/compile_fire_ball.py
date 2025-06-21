import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import joblib
import time
from datetime import datetime
from tqdm import tqdm

# ─────────── Config ───────────
CSV_FILE = "fire/modis_global_fire_2000_2025.csv"
CACHE_FILE = "fire/modis_balltree_cache.joblib"
R_EARTH_KM = 6371.0
NEWLINE = "\n"
# ──────────────────────────────

start_time = time.time()
print(f"⏳ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{NEWLINE}")

# ─────────── Load CSV ───────────
t0 = time.time()
print("📥 Loading MODIS fire CSV...")
df = pd.read_csv(CSV_FILE, usecols=["LATITUDE", "LONGITUDE", "ACQ_DATE", "ACQ_TIME"])
df = df.dropna(subset=["LATITUDE", "LONGITUDE", "ACQ_DATE", "ACQ_TIME"])
t1 = time.time()
print(f"✅ Loaded {len(df):,} fire records in {t1 - t0:.2f} sec{NEWLINE}")

# ───── Parse Timestamps ─────
print("🕒 Parsing timestamps (vectorized)...")
t2 = time.time()
df["ACQ_TIME"] = df["ACQ_TIME"].astype(str).str.zfill(4)
df["datetime_str"] = df["ACQ_DATE"] + df["ACQ_TIME"]
df["timestamp"] = pd.to_datetime(df["datetime_str"], format="%Y-%m-%d%H%M", errors="coerce").astype("int64") // 10**9
df = df.dropna(subset=["timestamp"])
t3 = time.time()
print(f"✅ Parsed timestamps and kept {len(df):,} rows in {t3 - t2:.2f} sec{NEWLINE}")

# ───── Convert Coordinates ─────
print("🧭 Converting coordinates to radians...")
t4 = time.time()
coords_rad = np.radians(df[["LATITUDE", "LONGITUDE"]].astype("float32").values)
t5 = time.time()
print(f"✅ Converted coordinates in {t5 - t4:.2f} sec{NEWLINE}")

# ───── Build BallTree ─────
print("🌳 Building BallTree...")
t6 = time.time()
tree = BallTree(coords_rad, metric="haversine")
t7 = time.time()
print(f"✅ BallTree built in {t7 - t6:.2f} sec{NEWLINE}")

# ───── Save Output ─────
print("💾 Saving cache...")
t8 = time.time()
joblib.dump((tree, df), CACHE_FILE)
t9 = time.time()
print(f"✅ Saved cache: {CACHE_FILE} (rows={len(df):,}) in {t9 - t8:.2f} sec{NEWLINE}")

# ───── Final Report ─────
total_time = time.time() - start_time
print(f"🏁 Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Total elapsed time: {total_time:.2f} sec")
