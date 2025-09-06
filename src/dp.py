import pandas as pd
from collections import defaultdict
from src.utils import (
    compute_sinr_and_rate,
    compute_score,
    update_m_s_t_from_channels
)

####################################################################################################
def run_dp_path_for_user(
    user_id: int,
    t_start: int,
    t_end: int,
    W: int,
    access_matrix: list,
    path_loss: dict,
    sat_channel_dict: dict,
    params: dict
):
    """
    DP + state counter，狀態為 (sat, ch)，保證 W slot 內 channel 不變
    """
    # 收集 t_start~t_end 可見的 (sat, ch) 狀態
    pairs_in_period = set()
    for t_abs in range(t_start, t_end + 1):
        for sat in access_matrix[t_abs]["visible_sats"]:
            for ch in sat_channel_dict[sat]:
                if sat_channel_dict[sat][ch] == 0:  # channel 可用
                    pairs_in_period.add((sat, ch))
    all_pairs = sorted(pairs_in_period)  # 固定順序
    P = len(all_pairs)
    T_len = t_end - t_start + 1
    NEG_INF = -1e18

    # dp[t_rel][p_idx][c] ,rel means relative to t_start
    dp = [[[NEG_INF] * (W + 1) for _ in range(P)] for _ in range(T_len)]
    parent = [[[None] * (W + 1) for _ in range(P)] for _ in range(T_len)]
    data_rate_cache = {}

    # 初始化 (t=0)
    for p_idx, (sat, ch) in enumerate(all_pairs):
        if sat not in access_matrix[t_start]["visible_sats"]:
            continue
        SINR, dr = compute_sinr_and_rate(params, path_loss, sat, t_start, sat_channel_dict, ch)
        if dr is None:
            continue
        m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
        score = compute_score(params, m_s_t, dr, sat)
        dp[0][p_idx][W] = score
        #回朔時候用來偵測parent是否=-1
        parent[0][p_idx][W] = (-1, -1, -1)
        data_rate_cache[(0, p_idx)] = dr

    # DP 遞推
    for t_rel in range(1, T_len):
        t_abs = t_start + t_rel
        for p_idx, (sat, ch) in enumerate(all_pairs):
            if sat not in access_matrix[t_abs]["visible_sats"]:
                continue
            SINR, dr = compute_sinr_and_rate(params, path_loss, sat, t_abs, sat_channel_dict, ch)
            if dr is None:
                continue
            m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
            score = compute_score(params, m_s_t, dr, sat)
            data_rate_cache[(t_rel, p_idx)] = dr
            ###能進入以下邏輯表示這個pair是可用的
            # 1. 換手 (c=0)
            best_val = NEG_INF
            best_parent = None
            for sp in range(P):
                if all_pairs[sp][0] == sat:  # 同衛星換 channel 不允許
                    continue
                if dp[t_rel-1][sp][W] != NEG_INF:
                    val = dp[t_rel-1][sp][W] + score
                    if val > best_val:
                        best_val = val
                        best_parent = (sp, t_rel-1, W)
            dp[t_rel][p_idx][0] = best_val
            parent[t_rel][p_idx][0] = best_parent

            # 2. 無換手（同一 (sat, ch)）
            if dp[t_rel-1][p_idx][0] != NEG_INF:
                dp[t_rel][p_idx][1] = dp[t_rel-1][p_idx][0] + score
                parent[t_rel][p_idx][1] = (p_idx, t_rel-1, 0)
            for c in range(2, W+1):
                candidates = []
                # 1️⃣ 從上一個 slot 的 c-1 狀態延續過來
                if dp[t_rel-1][p_idx][c-1] != NEG_INF:
                    candidates.append((dp[t_rel-1][p_idx][c-1], (p_idx, t_rel-1, c-1)))
                # 2️⃣ 或是直接從上一個 slot 的 c 狀態延續過來
                if dp[t_rel-1][p_idx][c] != NEG_INF:
                    candidates.append((dp[t_rel-1][p_idx][c], (p_idx, t_rel-1, c)))
                if candidates:
                    best_val2, best_parent2 = max(candidates, key=lambda x: x[0])
                    dp[t_rel][p_idx][c] = best_val2 + score
                    parent[t_rel][p_idx][c] = best_parent2

    # 找最大值 找dp[t_end][(s,c)][c]的最大值
    max_reward = NEG_INF
    end_state = None
    for p_idx in range(P):
        for c in range(W+1):
            if dp[T_len-1][p_idx][c] > max_reward:
                max_reward = dp[T_len-1][p_idx][c]
                end_state = (p_idx, T_len-1, c)

    if max_reward == NEG_INF:
        return [], 0, False, []

    # 回溯
    path = []
    data_rate_records = []
    cur = end_state
    #cur = (p_idx, t_rel, c)
    while cur and cur[0] != -1:
        p_idx, t_rel, c = cur
        sat, ch = all_pairs[p_idx]
        t_abs = t_start + t_rel
        dr = data_rate_cache.get((t_rel, p_idx), 0)
        path.append((sat, ch, t_abs))
        data_rate_records.append((user_id, t_abs, sat, ch, dr))
        # 回溯到上一個狀態
        cur = parent[t_rel][p_idx][c]

    path.reverse()
    return path, max_reward, True, data_rate_records


####################################################################################################
def run_dp_per_W(
    user_df: pd.DataFrame,
    access_matrix: list,
    path_loss: dict,
    sat_load_dict_backup: dict,
    params: dict,
    W: int = 2
):
    sat_load_dict = {sat: chs.copy() for sat, chs in sat_load_dict_backup.items()}
    user_df = user_df.sort_values(by="t_start").reset_index(drop=True)
    active_user_paths = []
    all_user_paths = []
    results = []
    load_by_time = defaultdict(lambda: defaultdict(int))
    all_user_data_rates = []

    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])

        # 釋放已完成的使用者
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                path = old_user.get("path")
                if path:
                    unique_sat_ch = set((s, c) for s, c, _ in path)
                    for sat, ch in unique_sat_ch:
                        sat_load_dict[sat][ch] = max(0, sat_load_dict[sat][ch] - 1)
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)

        # 計算路徑（DP）
        path, reward, success, data_rate_records = run_dp_path_for_user(
            user_id=user_id,
            t_start=t_start,
            t_end=t_end,
            W=W,
            access_matrix=access_matrix,
            path_loss=path_loss,
            sat_channel_dict=sat_load_dict,
            params=params
        )

        # 更新負載
        if path:
            unique_sat_ch = set((s, c) for s, c, _ in path)
            for sat, ch in unique_sat_ch:
                sat_load_dict[sat][ch] += 1
            for s, c, t in path:
                load_by_time[t][s] = load_by_time[t].get(s, 0) + 1

        all_user_data_rates.extend(data_rate_records)

        # 紀錄結果
        if success:
            active_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_begin": t_start,
                "t_end": t_end,
                "success": success,
                "reward": reward
            })

        all_user_paths.append({
            "user_id": user_id,
            "path": path,
            "t_begin": t_start,
            "t_end": t_end,
            "success": success,
            "reward": reward
        })

        results.append({
            "user_id": user_id,
            "reward": reward if success else None,
            "success": success
        })

    df_data_rates = pd.DataFrame(all_user_data_rates, columns=["user_id", "time", "sat", "channel", "data_rate"])
    return pd.DataFrame(results), all_user_paths, load_by_time, df_data_rates
