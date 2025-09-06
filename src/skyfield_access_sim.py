from skyfield.api import load, Topos, utc
from datetime import datetime, timedelta
import pandas as pd
import pickle

# === 參數設定 ===
TLE_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle'
USER_LAT = 40.0386
USER_LON = -75.5966
START_TIME_UTC = datetime(2025, 4, 21, 0, 0, 0, tzinfo=utc)

# ✅ 總測試時間 = 10 分鐘，每 slot = 15 秒
TOTAL_MINUTES = 10
SLOT_INTERVAL_SEC = 15
SLOTS = TOTAL_MINUTES * 60 // SLOT_INTERVAL_SEC  # 40

OUTPUT_CSV = "data/access_matrix.csv"
OUTPUT_PKL = "data/satellite_positions.pkl"
OUTPUT_NAMES = "data/all_satellite_names.csv"

# === 載入 TLE ===
print("Loading TLE data...")
satellites = load.tle_file(TLE_URL)
print(f"✅ Starlink 衛星總數：{len(satellites)} 顆")

ts = load.timescale()
observer = Topos(latitude_degrees=USER_LAT, longitude_degrees=USER_LON)

# ====== 模擬時間序列 ========
times = [ts.utc(START_TIME_UTC + timedelta(seconds=i * SLOT_INTERVAL_SEC)) for i in range(SLOTS)]

# ====== 可視性計算 ======
access_matrix = []
satellite_positions = {}

for t_idx, t in enumerate(times):
    visible_sats = []
    for sat in satellites:
        difference = sat - observer #得到位置向量
        topocentric = difference.at(t)
        alt, az, dist = topocentric.altaz()

        # ✅ Elevation 門檻 60 度
        if alt.degrees >= 60:
            visible_sats.append(sat.name)

        # 儲存每個 (sat.name, t_idx) 的位置 (x,y,z) km
        sat_xyz = sat.at(t).position.km.tolist()
        satellite_positions[(sat.name, t_idx)] = sat_xyz

    access_matrix.append({
        "time_slot": t_idx,
        "visible_sats": visible_sats
    })


# ====== 輸出檔案 ======
df = pd.DataFrame(access_matrix)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved access matrix to {OUTPUT_CSV}")

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(satellite_positions, f)
print(f"✅ Saved satellite_positions.pkl")

df_names = pd.DataFrame({"sat_name": [sat.name for sat in satellites]})
df_names.to_csv(OUTPUT_NAMES, index=False)
print(f"✅ Saved all_satellite_names.csv")
