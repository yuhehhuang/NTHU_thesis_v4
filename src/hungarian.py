from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import ast, math
from collections import defaultdict
from src.utils import compute_sinr_and_rate

def run_hungarian_per_W(df_users, df_access, path_loss, sat_channel_dict_backup, sat_positions, params, W):
    """
    匈牙利演算法（批次分配、W-slot 保留、按時間管理通道占用 + 每衛星同時上限 + 同一時刻分片配對）。
    - 新增：同一個時間 t 將候選使用者分成 per_t_slices 片（預設 2），逐片配對，每片之間更新占用/負載。
    - 沿用：busy_until 管理資源、每衛星上限 cap_limit、<=t 參賽、W-slot 可視/可用檢查、paths 全量輸出。
    """
    time_slots = int(df_access["time_slot"].max()) + 1 if "time_slot" in df_access.columns else len(df_access)
    alpha = params.get("alpha", 1.0)
    per_t_slices = max(1, int(params.get("per_t_slices", 2)))  # 👈 這裡控制同一時刻切幾片；預設 2

    # === busy-until 管理通道 + 承接背景占用 ===
    busy_until = {sat: {ch: -1 for ch in chs.keys()} for sat, chs in sat_channel_dict_backup.items()}
    for sat, chs in sat_channel_dict_backup.items():
        for ch, occ in chs.items():
            if occ:  # 背景占用視為整段期間忙
                busy_until[sat][ch] = time_slots - 1

    # 每顆衛星的同時上限（可用 params["max_channels_per_sat"] 覆寫）
    hard_cap = params.get("max_channels_per_sat", None)
    cap_limit = {
        sat: (min(len(chs), int(hard_cap)) if hard_cap is not None else len(chs))
        for sat, chs in busy_until.items()
    }

    def build_channel_snapshot_at(t_now: int):
        return {sat: {ch: 1 if until >= t_now else 0 for ch, until in chs.items()}
                for sat, chs in busy_until.items()}

    user_assignments = defaultdict(list)             # user_id -> [(t, sat, ch)]
    data_rate_records = []
    load_by_time = defaultdict(lambda: defaultdict(int))

    def sat_used_count_at_t(sat: str, t_now: int) -> int:
        return sum(1 for _, until in busy_until[sat].items() if until >= t_now)

    def sat_load_ratio_at_t(sat: str, t_now: int) -> float:
        used = sat_used_count_at_t(sat, t_now)
        total = len(busy_until[sat])
        return (used / total) if total > 0 else 0.0

    # 封裝：針對「某個子集使用者」在時刻 t 進行一次匈牙利配對（會修改 busy_until / records / next_available_time）
    def match_for_subset(t: int, subset_users: list, next_available_time: dict):
        if not subset_users:
            return set()

        # 取出 t 的可見衛星
        visible_sats_str = df_access[df_access["time_slot"] == t]["visible_sats"].iloc[0]
        visible_sats = ast.literal_eval(visible_sats_str) if isinstance(visible_sats_str, str) else visible_sats_str

        # 這一刻的佔用快照，提供給 SINR/干擾計算
        snapshot_t = build_channel_snapshot_at(t)

        # 預算 slots_remaining（每衛星同時上限 - 現用量）
        in_use_now = {sat: sat_used_count_at_t(sat, t) for sat in busy_until.keys()}
        slots_remaining = {sat: max(0, cap_limit[sat] - in_use_now.get(sat, 0)) for sat in busy_until.keys()}

        # 列出各衛星仍可用的 (ch, score)，取前 slots_remaining 條
        candidate_pairs = []
        pair_scores = {}

        for sat in visible_sats:
            if sat not in busy_until or slots_remaining.get(sat, 0) <= 0:
                continue

            free_ch_infos = []
            for ch, until in busy_until[sat].items():
                if until >= t:
                    continue
                if (sat, t) not in path_loss:
                    continue

                _, rate = compute_sinr_and_rate(params, path_loss, sat, t, snapshot_t, ch)
                # 0~1，越大表示越忙
                load_ratio = sat_load_ratio_at_t(sat, t)

                # 懲罰係數：滿載時最多打折 (1 - alpha)
                penalty = max(0.0, 1.0 - alpha * load_ratio)  # 夾一下避免浮點誤差

                score = rate * penalty

                free_ch_infos.append((ch, score, rate))

            if not free_ch_infos:
                continue

            free_ch_infos.sort(key=lambda x: x[1], reverse=True)
            k = int(slots_remaining[sat])
            for ch, score, rate in free_ch_infos[:k]:
                candidate_pairs.append((sat, ch))
                pair_scores[(sat, ch)] = {"score": score, "data_rate": rate}

        if not candidate_pairs:
            return set()

        # 匈牙利：cost matrix + W-slot 可見性/可用性檢查
        n_users = len(subset_users)
        n_pairs = len(candidate_pairs)
        cost_matrix = np.full((n_users, n_pairs), 1e9)

        for i, uid in enumerate(subset_users):
            t_end_user = int(df_users.loc[uid, "t_end"])
            last_used_sat = user_assignments[uid][-1][1] if user_assignments[uid] else None

            for j, (sat, ch) in enumerate(candidate_pairs):
                will_handover = (last_used_sat != sat)
                target_last_slot = min(t + W - 1, t_end_user) if will_handover else t

                feasible = True
                for future_t in range(t, target_last_slot + 1):
                    if future_t >= time_slots:
                        feasible = False
                        break
                    fut_vis_str = df_access[df_access["time_slot"] == future_t]["visible_sats"].iloc[0]
                    fut_vis = ast.literal_eval(fut_vis_str) if isinstance(fut_vis_str, str) else fut_vis_str
                    if sat not in fut_vis:
                        feasible = False
                        break
                    if busy_until[sat][ch] >= future_t:
                        feasible = False
                        break
                if not feasible:
                    continue

                cost_matrix[i, j] = -pair_scores[(sat, ch)]["score"]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_uids = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] > 1e8:
                continue

            uid = subset_users[i]
            sat, ch = candidate_pairs[j]
            info = pair_scores[(sat, ch)]
            t_end = int(df_users.loc[uid, "t_end"])

            last_used_sat = user_assignments[uid][-1][1] if user_assignments[uid] else None
            handover = (last_used_sat != sat)
            t_last = min(t + W - 1, t_end) if handover else t

            for t_used in range(t, t_last + 1):
                user_assignments[uid].append((t_used, sat, ch))
                data_rate_records.append({
                    "user_id": uid, "time": t_used, "sat": sat, "channel": ch,
                    "data_rate": info["data_rate"]
                })
                load_by_time[t_used][sat] += 1

            busy_until[sat][ch] = t_last
            next_available_time[uid] = t_last + 1
            if next_available_time[uid] > t_end:
                # 呼叫端會從 remaining_users 移除
                pass
            assigned_uids.add(uid)

        return assigned_uids

    # ===== 主流程 =====
    t_global = 0
    while t_global < time_slots:
        # 這一批：t_start == t_global 的使用者
        batch_users = df_users[df_users["t_start"] == t_global].index.tolist()
        if not batch_users:
            t_global += 1
            continue

        t = t_global
        next_available_time = {uid: t for uid in batch_users}
        remaining_users = set(batch_users)

        while remaining_users and t < time_slots:
            # 移除已過期者
            for uid in list(remaining_users):
                if t > int(df_users.loc[uid, "t_end"]):
                    remaining_users.remove(uid)

            if not remaining_users:
                break

            # 允許 <= t 的使用者參賽（避免飢餓）
            candidate_users = [
                uid for uid in remaining_users
                if next_available_time[uid] <= t and df_users.loc[uid, "t_end"] >= t
            ]
            if not candidate_users:
                t += 1
                continue

            # ---- 這裡做「分片」：把 candidate_users 切成 per_t_slices 份，逐片配對 ----
            ordered = sorted(candidate_users)  # 也可改成依剩餘需求、路徑長度排序
            assigned_this_t_total = set()
            for slice_idx in range(per_t_slices):
                start = math.ceil(len(ordered) *  slice_idx      / per_t_slices)
                end   = math.ceil(len(ordered) * (slice_idx + 1) / per_t_slices)
                sub = ordered[start:end]

                # 過濾出仍具資格（可能前片已分配、next_available_time 改變）
                sub = [uid for uid in sub if next_available_time[uid] <= t and df_users.loc[uid, "t_end"] >= t]
                if not sub:
                    continue

                got = match_for_subset(t, sub, next_available_time)
                assigned_this_t_total |= got

                # 將「任務完成」的從 remaining_users 移除
                for uid in list(got):
                    if next_available_time[uid] > int(df_users.loc[uid, "t_end"]):
                        remaining_users.discard(uid)

            # （安全檢查）當前 t 的 sat 使用不會超容量
            for sat in busy_until.keys():
                in_use = sat_used_count_at_t(sat, t)
                if in_use > cap_limit[sat]:
                    print(f"[WARN] t={t} sat={sat} 使用通道 {in_use} > 上限 {cap_limit[sat]}")

            # 進入下一個時間
            t += 1

        t_global += 1

    # === 輸出結果 ===
# === 輸出結果 ===
    df_results = pd.DataFrame(data_rate_records)

    # 預處理：把 user_id / data_rate 轉成數值，並先彙總每位使用者的總速率
    if not df_results.empty:
        df_results["user_id"]  = pd.to_numeric(df_results["user_id"], errors="coerce")
        df_results["data_rate"] = pd.to_numeric(df_results["data_rate"], errors="coerce").fillna(0.0)
        rate_lookup = df_results.groupby("user_id", dropna=True)["data_rate"].sum().to_dict()
    else:
        rate_lookup = {}

    # 一定包含所有 user（沒分配的也記錄 success=False）
    formatted_paths = []
    for uid in df_users.index:
        entries = sorted(user_assignments[uid], key=lambda x: x[0]) if user_assignments[uid] else []
        path_list = [(sat, ch, t) for (t, sat, ch) in entries]
        if entries:
            t_begin = entries[0][0]
            t_end_got = entries[-1][0]
        else:
            t_begin = int(df_users.loc[uid, "t_start"])
            t_end_got = np.nan

        target_span = int(df_users.loc[uid, "t_end"] - df_users.loc[uid, "t_start"] + 1)
        success = (len(entries) == target_span) and (len(entries) > 0) and \
                (entries[0][0] == int(df_users.loc[uid, "t_start"])) and \
                all(entries[k+1][0] == entries[k][0] + 1 for k in range(len(entries)-1))

        # ✅ 這裡不要再迭代 dict 了，直接查表或 0
        total_rate = float(rate_lookup.get(float(uid), rate_lookup.get(int(uid), 0.0)))

        formatted_paths.append([uid, str(path_list), t_begin, t_end_got, success, total_rate])

    df_paths = pd.DataFrame(formatted_paths,
                            columns=["user_id", "path", "t_begin", "t_end", "success", "reward"])


    # 方便觀察：回報沒拿到任何 slot 的使用者數量
    unassigned = [int(uid) for uid in df_users.index if len(user_assignments[uid]) == 0]
    if unassigned:
        print(f"[INFO] 未分配任何 slot 的使用者數量：{len(unassigned)}（前10個：{unassigned[:10]}）")

    return df_results, df_paths, load_by_time
