from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import ast
from collections import defaultdict
from src.utils import compute_sinr_and_rate

def compute_sat_load(channel_status_dict):
    """計算衛星負載比例：已使用頻道數 / 總頻道數"""
    total_channels = len(channel_status_dict)
    used_channels = sum(channel_status_dict.values())
    return used_channels / total_channels if total_channels > 0 else 0

def run_hungarian_per_W(df_users, df_access, path_loss, sat_channel_dict_backup, sat_positions, params, W):
    """
    使用匈牙利演算法 (Hungarian Algorithm) 進行批次使用者分配，
    保證每次換手後可維持至少 W 個 slot 連線（或直到該 user 的 t_end）。
    
    流程：
    1. 逐批處理 t_start 相同的使用者（batch-by-batch allocation）
    2. 這批使用者在分配完成前，不會插入新的使用者
    3. 當發生換手時，固定該衛星 W 個 slot；若未換手，僅分配當前 slot
    4. 在 cost matrix 建立階段，檢查未來 W-slot 的可見性
    """

    time_slots = len(df_access)
    alpha = params["alpha"]

    # === 初始化狀態 ===
    sat_load_dict = {sat: chs.copy() for sat, chs in sat_channel_dict_backup.items()}  # 衛星→channel 使用狀態
    user_assignments = defaultdict(list)  # user_id → [(t, sat, ch)]
    data_rate_records = []                # 所有分配紀錄 (for 輸出 data_rate.csv)
    load_by_time = defaultdict(lambda: defaultdict(int))  # t → {sat: 當前負載數}

    # === 外層時間迴圈：逐批處理 ===
    t_global = 0
    while t_global < time_slots:

        # Step 1: 找出這批 t_start == t_global 的使用者
        batch_users = df_users[df_users["t_start"] == t_global].index.tolist()
        if not batch_users:
            t_global += 1
            continue

        # Step 2: 初始化這批使用者的分配狀態
        t = t_global
        next_available_time = {uid: t for uid in batch_users}  # 每個 user 下一次可以參加 matching 的時間
        remaining_users = set(batch_users)  # 仍需分配的使用者

        # === 處理這批使用者直到全部分配完成 ===
        while len(remaining_users) > 0 and t < time_slots:

            # Step 3: 釋放已完成任務的使用者資源
            for uid in list(user_assignments):
                t_end = df_users.loc[uid, "t_end"]
                if t == t_end + 1:  # 該 user 已結束
                    for _, sat, ch in user_assignments[uid]:
                        sat_load_dict[sat][ch] = 0  # 該 channel 釋放

            # Step 4: 過濾出這批中目前可參與分配的使用者
            candidate_users = [
                uid for uid in remaining_users
                if next_available_time[uid] == t and df_users.loc[uid, "t_end"] >= t
            ]
            if not candidate_users:
                t += 1
                continue

            # Step 5: 取得目前 t 可見的衛星清單（全域可見衛星）
            visible_sats_str = df_access[df_access["time_slot"] == t]["visible_sats"].iloc[0]
            visible_sats = ast.literal_eval(visible_sats_str) if isinstance(visible_sats_str, str) else visible_sats_str

            # Step 6: 建立當前可用的 (sat, ch) 清單
            candidate_pairs = []
            pair_scores = {}
            for sat in visible_sats:
                if sat not in sat_load_dict:
                    continue
                for ch, occupied in sat_load_dict[sat].items():
                    if occupied != 0:
                        continue
                    if (sat, t) not in path_loss:
                        continue
                    # 先計算一個基礎分數（暫時不做 W-slot 檢查）
                    _, rate = compute_sinr_and_rate(params, path_loss, sat, t, sat_load_dict, ch)
                    load_score = 1 - compute_sat_load(sat_load_dict[sat])
                    score = load_score * rate * alpha
                    candidate_pairs.append((sat, ch))
                    pair_scores[(sat, ch)] = {
                        "score": score,
                        "data_rate": rate
                    }

            if not candidate_pairs:
                t += 1
                continue

            # Step 7: 建立 cost matrix，針對每個 (user, sat, ch) 檢查 W-slot 可見性
            n_users = len(candidate_users)
            n_pairs = len(candidate_pairs)
            cost_matrix = np.full((n_users, n_pairs), 1e9)  # 預設不可行

            for i, uid in enumerate(candidate_users):
                t_end_user = df_users.loc[uid, "t_end"]
                for j, (sat, ch) in enumerate(candidate_pairs):
                    # 檢查換手後 W-slot 的可見性
                    last_used_sat = user_assignments[uid][-1][1] if user_assignments[uid] else None
                    will_handover = (last_used_sat != sat)
                    if will_handover:
                        target_last_slot = min(t + W - 1, t_end_user)
                    else:
                        target_last_slot = t  # 非換手只檢查當下

                    feasible = True
                    for future_t in range(t, target_last_slot + 1):
                        if future_t >= time_slots:
                            feasible = False
                            break
                        future_visible_str = df_access[df_access["time_slot"] == future_t]["visible_sats"].iloc[0]
                        if isinstance(future_visible_str, str):
                            future_visible = ast.literal_eval(future_visible_str)
                        else:
                            future_visible = future_visible_str
                        if sat not in future_visible:
                            feasible = False
                            break
                    if not feasible:
                        continue  # 保持 cost=1e9

                    # 有效 → 使用預計算的 score
                    cost_matrix[i, j] = -pair_scores[(sat, ch)]["score"]

            # 匈牙利演算法求解
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Step 8: 根據配對結果進行實際分配
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] > 1e8:  # 不可行
                    continue

                uid = candidate_users[i]
                sat, ch = candidate_pairs[j]
                info = pair_scores[(sat, ch)]
                t_end = df_users.loc[uid, "t_end"]

                # 判斷是否換手，決定要固定多久
                last_used_sat = user_assignments[uid][-1][1] if user_assignments[uid] else None
                handover = (last_used_sat != sat)
                t_last = min(t + W - 1, t_end) if handover else t

                # 實際記錄分配結果
                for t_used in range(t, t_last + 1):
                    sat_load_dict[sat][ch] = 1
                    user_assignments[uid].append((t_used, sat, ch))
                    data_rate_records.append({
                        "user_id": uid,
                        "time": t_used,
                        "sat": sat,
                        "channel": ch,
                        "data_rate": info["data_rate"]
                    })
                    load_by_time[t_used][sat] += 1

                # 更新下一次可分配時間
                next_available_time[uid] = t_last + 1
                if next_available_time[uid] > t_end:
                    remaining_users.remove(uid)

            t += 1  # 處理這批 user 的下一個時間

        # Step 9: 批次完成，前進到下一批使用者
        t_global += 1

    # === 輸出結果 ===
    df_results = pd.DataFrame(data_rate_records)

    # 轉換成 greedy 相同格式的 all_user_paths
    formatted_paths = []
    for uid, entries in user_assignments.items():
        if not entries:
            continue
        entries = sorted(entries, key=lambda x: x[0])
        path_list = [(sat, ch, t) for (t, sat, ch) in entries]
        t_begin = entries[0][0]
        t_end = entries[-1][0]
        success = (t_end - t_begin + 1) == (df_users.loc[uid, "t_end"] - df_users.loc[uid, "t_start"] + 1)
        total_rate = sum(d["data_rate"] for d in data_rate_records if d["user_id"] == uid)
        formatted_paths.append([uid, str(path_list), t_begin, t_end, success, total_rate])

    df_paths = pd.DataFrame(formatted_paths, columns=["user_id", "path", "t_begin", "t_end", "success", "reward"])

    return df_results, df_paths, load_by_time