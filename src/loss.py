import pandas as pd
import pickle
import numpy as np
from math import radians, cos, sin, atan2, degrees

# === 常數 ===
FC = 14.5e9           # Ka-band 中心頻率
C_LIGHT = 3e8
EARTH_RADIUS = 6371.0
USER_LAT = 40.0386    # Target Area 中心
USER_LON = -75.5966

# 3GPP TR 38.811 Table 6.6.2-1 (Ka-band LOS)
KA_BAND_LOS_SIGMA = {10: 2.9, 20: 2.4, 30: 2.7, 40: 2.4, 50: 2.4,
                     60: 2.7, 70: 2.6, 80: 2.8, 90: 0.6}

# === 工具函式 ===
def get_sigma_sf(elev):
    """根據仰角找對應 shadow fading σSF"""
    return KA_BAND_LOS_SIGMA[min(KA_BAND_LOS_SIGMA.keys(), key=lambda x: abs(x - elev))]

def fspl(distance_km, fc):
    """自由空間路徑損耗 (linear)"""
    return (4 * np.pi * distance_km * 1000 * fc / C_LIGHT) ** 2

def geodetic_to_ecef(lat, lon, alt_km=0):
    """地理座標轉換成 ECEF"""
    lat_r, lon_r = radians(lat), radians(lon)
    R = EARTH_RADIUS + alt_km
    return np.array([
        R * cos(lat_r) * cos(lon_r),
        R * cos(lat_r) * sin(lon_r),
        R * sin(lat_r)
    ])

def compute_elevation(user_ecef, sat_ecef):
    """計算仰角 (degrees)"""
    user_norm = user_ecef / np.linalg.norm(user_ecef)
    sat_vec = sat_ecef - user_ecef
    sat_norm = sat_vec / np.linalg.norm(sat_vec)
    angle = degrees(atan2(np.linalg.norm(np.cross(user_norm, sat_norm)), np.dot(user_norm, sat_norm)))
    return 90 - angle

# === 主程式 ===
def compute_path_loss(user_lat=USER_LAT, user_lon=USER_LON):
    # 讀取資料
    df_access = pd.read_csv("data/access_matrix.csv")
    df_access["visible_sats"] = df_access["visible_sats"].apply(eval)

    with open("data/satellite_positions.pkl", "rb") as f:
        sat_positions = pickle.load(f)

    user_ecef = geodetic_to_ecef(user_lat, user_lon)

    PL_dict = {}
    count = 0

    for t, row in df_access.groupby("time_slot"):
        visible_sats = row.iloc[0]["visible_sats"]

        for sat_name in visible_sats:
            sat_ecef = sat_positions.get((sat_name, t))
            if sat_ecef is None:
                continue

            d_km = np.linalg.norm(np.array(sat_ecef) - user_ecef)
            elevation = compute_elevation(user_ecef, np.array(sat_ecef))

            FSPL_linear = fspl(d_km, FC)

            sigma_sf = get_sigma_sf(min(90, max(10, elevation)))
            sf_linear = 10 ** (np.random.normal(0, sigma_sf) / 10)

            PL_dict[(sat_name, t)] = FSPL_linear * sf_linear
            count += 1

    print(f"✅ Path Loss 計算完成，共 {count} 筆資料")

    with open("data/path_loss.pkl", "wb") as f:
        pickle.dump(PL_dict, f)

# === 執行 ===
if __name__ == "__main__":
    compute_path_loss()
###
#最後得到的 PL_dict 是：
#{
#  ("SAT_A", 0): 1.23e22,
#  ("SAT_B", 0): 5.67e21,
#  ("SAT_B", 1): 6.89e21,
#  ("SAT_C", 1): 2.34e22,
#  ...
#}
###
