import numpy as np
from collections import defaultdict
import pandas as pd

def update_m_s_t_from_channels(sat_channel_dict, all_sats):
    """根據 channel 使用狀態計算每顆衛星的總負載"""
    #把satellite 此時刻使用的 channel 數量加總
    return {sat: sum(sat_channel_dict[sat].values()) for sat in all_sats}


def check_visibility(df_access, sat, t_start, t_end):
    """檢查衛星在 t_start~t_end 是否連續可見"""
    for t in range(t_start, t_end + 1):
        row = df_access[df_access["time_slot"] == t]
        if row.empty or sat not in row["visible_sats"].values[0]:
            return False
    return True

def compute_sinr_and_rate(params, path_loss, sat, t, sat_channel_dict, chosen_channel):
    PL = path_loss.get((sat, t))
    if PL is None:
        return None, None

    P_rx = params["eirp_linear"] * params["grx_linear"] / PL

    interference = 0
    for other_sat, channels in sat_channel_dict.items():
        if other_sat == sat:
            continue
        if channels.get(chosen_channel, 0) == 1:
            PL_other = path_loss.get((other_sat, t))
            if PL_other:
                interference += params["eirp_linear"] * params["grx_linear"] / PL_other

    SINR = P_rx / (params["noise_watt"] + interference)
    data_rate = params["channel_bandwidth_hz"] * np.log2(1 + SINR) /1e6  # Mbps
    return SINR, data_rate
    
def compute_score(params, m_s_t, data_rate, sat):
    L = m_s_t[sat] / params["num_channels"]
    return (1 - params["alpha"] * L) * data_rate

#這是每個user都分配完channel後正式去算他們彼此的干擾
def recompute_all_data_rates(all_user_paths, path_loss, params, sat_channel_dict_backup):
    """
    重新計算每個 user 的 data rate，考慮背景干擾 + 所有 user 的分配。
    """
    from collections import defaultdict
    import copy

    # 1️⃣ 先複製初始化的背景使用者狀態
    sat_channel_dict = copy.deepcopy(sat_channel_dict_backup)

    # 2️⃣ 建立一個 dict 記錄每個 time slot 的使用者分配
    assignments_by_time = defaultdict(list)  # {time: [(user_id, sat, ch), ...]}
    for entry in all_user_paths:
        user_id = entry["user_id"]
        path = entry["path"]
        if isinstance(path, str):
            try:
                path = eval(path)
            except:
                continue  # 如果 eval 出錯就跳過       
        for sat, ch, t in path:
            assignments_by_time[t].append((user_id, sat, ch))

    all_records = []

    # 3️⃣ 逐個 time slot 重新計算干擾 + data rate
    for t, assignments in assignments_by_time.items():
        # 先把這個 time slot 的使用者佔用標記到 sat_channel_dict
        temp_dict = copy.deepcopy(sat_channel_dict)
        for _, sat, ch in assignments:
            temp_dict[sat][ch] = 1  # 表示這個 slot 這個 channel 有 user 使用

        # 計算每個 user 的 SINR & data rate
        for user_id, sat, ch in assignments:
            SINR, data_rate = compute_sinr_and_rate(params, path_loss, sat, t, temp_dict, ch)
            all_records.append({
                "user_id": user_id,
                "time": t,
                "sat": sat,
                "channel": ch,
                "data_rate": data_rate if data_rate else 0
            })

    return pd.DataFrame(all_records)
