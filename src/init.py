import pandas as pd
import pickle
import json
import os



def init_sat_channel_all(satellites, num_channels=25, randomize=True, max_background_users=2):
    """
    初始化每顆衛星每個 channel 的使用狀態
    return: dict[sat_name][channel] = 0/1
    """
    import random
    sat_channel_dict = {}

    for sat in satellites:
        sat_channel_dict[sat] = {ch: 0 for ch in range(num_channels)}

        if randomize:
            used_channels = random.sample(
                range(num_channels),
                k=random.randint(0, min(max_background_users, num_channels))
            )
            for ch in used_channels:
                sat_channel_dict[sat][ch] = 1

    return sat_channel_dict


def load_system_data(regenerate_sat_channels=False,
                     sat_channel_path="data/sat_channel_dict_backup.pkl",
                     user_csv_path=None):
    # 支援外部指定 user csv 路徑
    # 優先順序：
    #  1) user_csv_path 參數（若呼叫時傳入）
    #  2) 環境變數 USER_CSV（如果你想用 env 指定）
    #  3) 預設 data/user_info.csv（原本行為）
    if user_csv_path is None:
        user_csv_path = os.environ.get("USER_CSV", "data/user_info.csv")

    if not os.path.exists(user_csv_path):
        raise FileNotFoundError(f"找不到 user CSV: {user_csv_path}. 請先用 user.py 產生，或指定 user_csv_path 參數。")

    # === 讀取 User 資料 ===
    df_users = pd.read_csv(user_csv_path)

    # === 讀取 Access Matrix ===
    df_access = pd.read_csv("data/access_matrix.csv")
    df_access["visible_sats"] = df_access["visible_sats"].apply(eval)

    # === 讀取衛星座標 ===
    with open("data/satellite_positions.pkl", "rb") as f:
        sat_positions = pickle.load(f)

    # === 讀取 Path Loss ===
    with open("data/path_loss.pkl", "rb") as f:
        path_loss = pickle.load(f)

    # === 讀取系統參數 ===
    with open("data/system_params.json", "r") as f:
        params = json.load(f)

    # ✅ 轉換參數
    params["eirp_linear"] = 10 ** (params["eirp_dbw"] / 10)
    params["grx_linear"] = 10 ** (params["grx_dbi"] / 10)
    params["noise_watt"] = (
        params["boltzmann"] * params["noise_temperature_k"] * params["channel_bandwidth_hz"]
    )

    # === 收集所有衛星名稱 ===
    all_satellites = set()
    for sats in df_access["visible_sats"]:
        all_satellites.update(sats)

    # === 載入或初始化衛星頻道狀態 ===
    if not regenerate_sat_channels and os.path.exists(sat_channel_path):
        with open(sat_channel_path, "rb") as f:
            sat_channel_dict_backup = pickle.load(f)
        print("✅ 已載入現有 sat_channel_dict_backup.pkl")
    else:
        sat_channel_dict_backup = init_sat_channel_all(
            satellites=all_satellites,
            num_channels=params["num_channels"],
            randomize=True,
            max_background_users=2
        )
        with open(sat_channel_path, "wb") as f:
            pickle.dump(sat_channel_dict_backup, f)
        print("✅ 重新初始化並儲存 sat_channel_dict_backup.pkl")

    system = {
        "users": df_users,
        "access_matrix": df_access,
        "sat_positions": sat_positions,
        "path_loss": path_loss,
        "params": params,
        "sat_channel_dict_backup": sat_channel_dict_backup
    }

    print("✅ System Data Loaded")
    print(f"Users: {len(df_users)}")
    print(f"Time Slots: {len(df_access)}")
    print(f"Sat Positions: {len(sat_positions)} entries")
    print(f"Path Loss entries: {len(path_loss)}")
    print(f"Sat Channels Loaded: {len(sat_channel_dict_backup)} satellites")

    return system

if __name__ == "__main__":
    # ✅ 預設為 False → 直接讀檔
    # ✅ 設為 True → 重新初始化並覆蓋
    system = load_system_data(regenerate_sat_channels=False)
