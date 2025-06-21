import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import joblib
import json
import time
from datetime import datetime
from tqdm import tqdm

# ─────────── Config ───────────
JSONL_FILE = "wind_cache_global.jsonl"
CACHE_FILE = "wind/wind_balltree_cache.joblib"
R_EARTH_KM = 6371.0
NEWLINE = "\n"
# ──────────────────────────────

start_time = time.time()
print(f"⏳ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{NEWLINE}")

# ───── Load JSONL Data ─────
print("📥 Loading wind JSONL file...")
t0 = time.time()
records = []
with open(JSONL_FILE, "r") as f:
    for line in f:
        try:
            obj = json.loads(line)
            lat_str, lon_str = obj["key"].split(",")
            lat = float(lat_str)
            lon = float(lon_str)
            ws = obj.get("ws")
            wd = obj.get("wd")
            if ws is not None and wd is not None:
                records.append({"latitude": lat, "longitude": lon, "wind_speed": ws, "wind_dir": wd})
        except Exception as e:
            print(f"⚠️ Error loading line: {e}")

df = pd.DataFrame(records)
t1 = time.time()
print(f"✅ Loaded {len(df):,} wind records in {t1 - t0:.2f} sec{NEWLINE}")

# ───── Convert Coordinates ─────
print("🧭 Converting coordinates to radians...")
t2 = time.time()
coords_rad = np.radians(df[["latitude", "longitude"]].astype("float32").values)
t3 = time.time()
print(f"✅ Converted coordinates in {t3 - t2:.2f} sec{NEWLINE}")

# ───── Build BallTree ─────
print("🌳 Building BallTree...")
t4 = time.time()
tree = BallTree(coords_rad, metric="haversine")
t5 = time.time()
print(f"✅ BallTree built in {t5 - t4:.2f} sec{NEWLINE}")

# ───── Save Output ─────
print("💾 Saving cache...")
t6 = time.time()
joblib.dump((tree, df), CACHE_FILE)
t7 = time.time()
print(f"✅ Saved cache: {CACHE_FILE} (rows={len(df):,}) in {t7 - t6:.2f} sec{NEWLINE}")

# ───── Final Report ─────
total_time = time.time() - start_time
print(f"🏁 Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"⏱️ Total elapsed time: {total_time:.2f} sec")
