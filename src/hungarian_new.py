from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import ast
from collections import defaultdict
from src.utils import compute_sinr_and_rate

def compute_sat_load(channel_status_dict):
    """計算衛星目前的負載比例（已用頻道 / 總頻道）"""
    total_channels = len(channel_status_dict)
    used_channels = sum(channel_status_dict.values())
    return used_channels / total_channels if total_channels > 0 else 0

def run_hungarian_new_per_W(
    df_users, df_access, path_loss, sat_channel_dict_backup, sat_positions, params, W,
    cap_per_sat=2  # ⭐ 每顆衛星同一時間最多可分配給 cap_per_sat 個 user（折衷參數）
):
    time_slots = len(df_access)
    alpha = params["alpha"]

    # 狀態容器
    sat_load_dict = {sat: chs.copy() for sat, chs in sat_channel_dict_backup.items()}
    user_assignments = defaultdict(list)              # uid → [(t, sat, ch)]
    data_rate_records = []                            # 記錄 (user_id, time, sat, channel, data_rate)
    load_by_time = defaultdict(lambda: defaultdict(int))  # t → {sat: 當下負載數}

    t_global = 0
    while t_global < time_slots:
        # ① 這批同時進場的使用者
        batch_users = df_users[df_users["t_start"] == t_global].index.tolist()
        if not batch_users:
            t_global += 1
            continue

        t = t_global
        next_available_time = {uid: t for uid in batch_users}
        remaining_users = set(batch_users)

        # 直到這批全部分配完成
        while remaining_users and t < time_slots:
            # ② 釋放已結束使用者佔用的頻道
            for uid in list(user_assignments):
                t_end = df_users.loc[uid, "t_end"]
                if t == t_end + 1:
                    for _, sat, ch in user_assignments[uid]:
                        sat_load_dict[sat][ch] = 0  # 釋放

            # ③ 篩出此刻能被處理的使用者（還在服務區間且到了自己的下一次匹配時點）
            candidate_users = [
                uid for uid in remaining_users
                if next_available_time[uid] == t and df_users.loc[uid, "t_end"] >= t
            ]
            if not candidate_users:
                t += 1
                continue

            # ④ 取得此刻可見衛星
            visible_sats_raw = df_access[df_access["time_slot"] == t]["visible_sats"].iloc[0]
            # data processing
            visible_sats = ast.literal_eval(visible_sats_raw) if isinstance(visible_sats_raw, str) else visible_sats_raw

            # ⑤ 建立「衛星名額」：每顆衛星挑出「分數最高的 cap_per_sat 個可用 channel」
            #    並把每個名額視為一個列（column）供匈牙利分配。
            #    sat_slots: [(sat, slot_idx)]，每個 slot_idx 對應一個不同 channel
            sat_slots = []
            slot_info = {}  # (sat, slot_idx) → (ch, score, rate)

            for sat in visible_sats:
                if sat not in sat_load_dict:
                    continue

                # 列出此刻 sat 的所有可用 channel 的 (ch, score, rate)
                cand = []
                for ch, occupied in sat_load_dict[sat].items():
                    if occupied != 0:
                        continue
                    if (sat, t) not in path_loss:
                        continue

                    # 計算該 (sat, ch) 此刻的 throughput 與負載分數
                    _, rate = compute_sinr_and_rate(params, path_loss, sat, t, sat_load_dict, ch)
                    load_score = 1 - compute_sat_load(sat_load_dict[sat])
                    score = load_score * rate * alpha
                    cand.append((ch, score, rate))

                if not cand:
                    continue

                # 只留「分數最高的 cap_per_sat 個」⇒ 形成最多 cap_per_sat 個“名額”
                cand.sort(key=lambda x: x[1], reverse=True)
                top_k = cand[:cap_per_sat]

                for idx, (ch, score, rate) in enumerate(top_k):
                    sat_slots.append((sat, idx))          # idx 是該衛星的第 idx 個名額
                    slot_info[(sat, idx)] = (ch, score, rate)

            if not sat_slots:
                t += 1
                continue

            # ⑥ 建立 cost matrix：列是 sat_slots（容量名額），行是 user
            n_users = len(candidate_users)
            n_slots = len(sat_slots)
            cost_matrix = np.full((n_users, n_slots), 1e9)

            # 填成本：可見性 + W-slot 可行性 + 分數（負號當成本）
            for i, uid in enumerate(candidate_users):
                t_end_user = df_users.loc[uid, "t_end"]
                last_used_sat = user_assignments[uid][-1][1] if user_assignments[uid] else None

                for j, (sat, slot_idx) in enumerate(sat_slots):
                    # 換手 ⇒ 要檢查 min(t+W-1, t_end_user) 內的可見性；未換手只需檢查當前 t
                    will_handover = (last_used_sat != sat)
                    target_last_slot = min(t + W - 1, t_end_user) if will_handover else t

                    feasible = True
                    for future_t in range(t, target_last_slot + 1):
                        if future_t >= time_slots:
                            feasible = False
                            break
                        future_vis_raw = df_access[df_access["time_slot"] == future_t]["visible_sats"].iloc[0]
                        future_visible = ast.literal_eval(future_vis_raw) if isinstance(future_vis_raw, str) else future_vis_raw
                        #只要時間內一次偵測到衛星不可見，就return false;
                        if sat not in future_visible:
                            feasible = False
                            break
                    if not feasible:
                        continue

                    # 使用該名額對應的分數作為效益；成本取負
                    _, score, _ = slot_info[(sat, slot_idx)]
                    cost_matrix[i, j] = -score

            # ⑦ 匈牙利分配（保證每 user 至多拿到 1 名額；每衛星最多 cap_per_sat 名額）
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # ⑧ 實際配置與記錄
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] > 1e8:
                    continue  # 此配對不可行

                uid = candidate_users[i]
                sat, slot_idx = sat_slots[j]
                ch, _, rate = slot_info[(sat, slot_idx)]
                t_end = df_users.loc[uid, "t_end"]

                # 判斷是否換手，決定要固定多久
                last_used_sat = user_assignments[uid][-1][1] if user_assignments[uid] else None
                handover = (last_used_sat != sat)
                t_last = min(t + W - 1, t_end) if handover else t

                # 連續占用（確保 W-slot 穩定或到 t_end）
                for t_used in range(t, t_last + 1):
                    sat_load_dict[sat][ch] = 1
                    user_assignments[uid].append((t_used, sat, ch))
                    data_rate_records.append({
                        "user_id": uid,
                        "time": t_used,
                        "sat": sat,
                        "channel": ch,
                        "data_rate": rate
                    })
                    load_by_time[t_used][sat] += 1

                # 更新此 user 下一次參賽時間
                next_available_time[uid] = t_last + 1
                if next_available_time[uid] > t_end:
                    remaining_users.remove(uid)

            # 進入下一個時間
            t += 1

        # 批次結束，往下一個 t_start
        t_global += 1

    # === 整理輸出 ===
    df_results = pd.DataFrame(data_rate_records)

    formatted_paths = []
    for uid, entries in user_assignments.items():
        if not entries:
            continue
        entries.sort(key=lambda x: x[0])
        path_list = [(sat, ch, t) for (t, sat, ch) in entries]
        t_begin = entries[0][0]
        t_end = entries[-1][0]
        success = (t_end - t_begin + 1) == (df_users.loc[uid, "t_end"] - df_users.loc[uid, "t_start"] + 1)
        total_rate = sum(d["data_rate"] for d in data_rate_records if d["user_id"] == uid)
        formatted_paths.append([uid, str(path_list), t_begin, t_end, success, total_rate])

    df_paths = pd.DataFrame(formatted_paths, columns=["user_id", "path", "t_begin", "t_end", "success", "reward"])
    return df_results, df_paths, load_by_time
