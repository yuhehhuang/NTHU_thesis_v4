# mslb.py
import networkx as nx
import copy
from src.utils import compute_sinr_and_rate, compute_score, update_m_s_t_from_channels
import pandas as pd
from collections import defaultdict

#去看t時間visible的衛星集合
def _visible_pairs_at_t(user_id, t_abs, access_matrix, sat_channel_dict):
    vis = []
    visible_sats = access_matrix[t_abs].get("visible_sats_for_user", {}).get(
        user_id, access_matrix[t_abs]["visible_sats"]
    )
    for s in visible_sats:
        for ch, used in sat_channel_dict[s].items():
            if used == 0:
                vis.append((s, ch))
    return vis

def _run_length(user_id, s, ch, t0, t_end, access_matrix, sat_channel_dict):
    """從 t0 開始，用 (s,ch) 的最長連續可用長度 L。"""
    L = 0
    for t in range(t0, t_end + 1):
        # 可見 & channel 空閒 才能延續
        visible = s in (
            access_matrix[t].get("visible_sats_for_user", {}).get(user_id, set())
            if "visible_sats_for_user" in access_matrix[t]
            else access_matrix[t]["visible_sats"]
        )
        if not visible or sat_channel_dict[s][ch] != 0:
            break
        L += 1
    return L

def _segment_score(user_id, s, ch, t0, L, params, path_loss, sat_channel_dict):
    """把整段 L 個 slot 的分數加總"""
    total = 0.0
    # 注意：這裡用“當下佔用狀態”的 m_s_t（和 dp.py 一致）
    m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
    for t in range(t0, t0 + L):
        _, dr = compute_sinr_and_rate(params, path_loss, s, t, sat_channel_dict, ch)
        if dr is None:
            dr = 0.0
        total += compute_score(params, m_s_t, dr, s)
    return total

def build_graph_for_user_segments(
    user_id: int,
    t_begin: int,
    t_end: int,
    W: int,
    access_matrix: list,
    path_loss: dict,
    sat_channel_dict: dict,
    params: dict,
    Lambda: float = 1e9,
):
    G = nx.DiGraph()
    G.add_node("START"); G.add_node("END")

    # 1) 產生 segments（不加邊）
    segments = []
    for t0 in range(t_begin, t_end + 1):
        min_L = 1 if t0 == t_begin else W
        for (s, ch) in _visible_pairs_at_t(user_id, t0, access_matrix, sat_channel_dict):
            L = _run_length(user_id, s, ch, t0, t_end, access_matrix, sat_channel_dict)
            if L >= min_L:
                segments.append((s, ch, t0, L))
                G.add_node(("SEG", s, ch, t0, L))

    # 2) 立刻早退檢查（放這裡！）
    has_start_seg = any(t0 == t_begin for _, _, t0, L in segments)
    has_end_seg   = any(t0 + L - 1 == t_end for _, _, t0, L in segments)
    if not has_start_seg or not has_end_seg:
        # 無法從 START 連到 END，直接回一張空圖
        G = nx.DiGraph(); G.add_node("START"); G.add_node("END")
        return G

    # 3) START → 起始段
    for (s, ch, t0, L) in segments:
        if t0 != t_begin:
            continue
        seg_score = _segment_score(user_id, s, ch, t0, L, params, path_loss, sat_channel_dict)
        G.add_edge("START", ("SEG", s, ch, t0, L), weight=Lambda - seg_score)

    # 4) 段與段之間
    index_by_t0 = {}
    for s, ch, t0, L in segments:
        index_by_t0.setdefault(t0, []).append((s, ch, L))

    for (s1, ch1, t0, L) in segments:
        t_next = t0 + L
        if t_next > t_end: 
            continue
        for (s2, ch2, L2) in index_by_t0.get(t_next, []):
            if s2 == s1:
                continue
            seg_score = _segment_score(user_id, s2, ch2, t_next, L2, params, path_loss, sat_channel_dict)
            G.add_edge(("SEG", s1, ch1, t0, L), ("SEG", s2, ch2, t_next, L2), weight=Lambda - seg_score)

    # 5) 覆蓋到 t_end 的段 → END
    for (s, ch, t0, L) in segments:
        if t0 + L - 1 == t_end:
            G.add_edge(("SEG", s, ch, t0, L), "END", weight=0.0)

    return G

def run_mslb_batch(
    user_df,
    access_matrix: list,
    path_loss: dict,
    sat_channel_dict: dict,
    params: dict,
    W: int
):
    user_df = user_df.sort_values(by="t_start").reset_index(drop=True)
    T = len(access_matrix)

    results = []
    all_user_paths = []
    all_user_data_rates = []
    load_by_time = defaultdict(lambda: defaultdict(int))

    def solve_one_user(u):
        t_begin = int(user_df.loc[user_df["user_id"] == u, "t_start"].iloc[0])
        t_end   = int(user_df.loc[user_df["user_id"] == u, "t_end"].iloc[0])

        # 用「當前」sat_channel_dict 建 segment 圖（不是快照）
        G = build_graph_for_user_segments(
            user_id=u,
            t_begin=t_begin,
            t_end=t_end,
            W=W,
            access_matrix=access_matrix,
            path_loss=path_loss,
            sat_channel_dict=sat_channel_dict,
            params=params,
        )

        # 最短路
        dist, paths = nx.single_source_dijkstra(G, "START", weight="weight")
        if "END" not in paths or not paths["END"]:
            return [], 0.0, [], t_begin, t_end

        path = paths["END"]

        # 計分 & 收集 data rate（沿用你的計分邏輯）
        total_reward = 0.0
        data_rate_records = []
        for node in path:
            if node in ("START", "END"):
                continue
            _, s, c, t0, L = node  # ("SEG", s, c, t0, L)
            for t in range(t0, t0 + L):
                m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
                _, dr = compute_sinr_and_rate(params, path_loss, s, t, sat_channel_dict, c)
                if dr is None:
                    dr = 0.0
                score = compute_score(params, m_s_t, dr, s)
                total_reward += score
                data_rate_records.append((u, t, s, c, dr))

        return path, total_reward, data_rate_records, t_begin, t_end

    total_users = len(user_df)

    for t in range(T):
        # 1) 釋放已完成的用戶（把他用過的 (s,c) 減 1）
        for entry in list(all_user_paths):
            if t == entry["t_end"] + 1:
                used_pairs = {(s, c) for (s, c, tt) in entry["path"]}
                for (s, c) in used_pairs:
                    sat_channel_dict[s][c] = max(0, sat_channel_dict[s][c] - 1)

        # 2) 這一時刻新進入的 users，依序處理
        entrants = user_df[user_df["t_start"] == t]["user_id"].tolist()
        if not entrants:
            continue

        for u in entrants:
            print(f"[time {t}] processing user {u} ...")

            # 2a) 算路徑（在當前 sat_channel_dict 下）
            path, reward, recs, t_b, t_e = solve_one_user(u)

            # 2b) 展開成 (s, c, t)；立刻更新衛星狀態與 load_by_time
            expanded_path = []
            used_pairs = set()
            for node in path:
                if node in ("START", "END"):
                    continue
                _, s, c, t0, L = node
                used_pairs.add((s, c))
                for tt in range(t0, t0 + L):
                    expanded_path.append((s, c, tt))
                    load_by_time[tt][s] += 1

            # 只需對每個 (s,c) +1（代表該使用者在整段服務期佔用這條 channel）
            for (s, c) in used_pairs:
                sat_channel_dict[s][c] += 1

            # 2c) 記錄輸出
            all_user_data_rates.extend(recs)
            all_user_paths.append({
                "user_id": u,
                "path": expanded_path,   # [(s,c,t), ...]
                "t_begin": t_b,
                "t_end": t_e,
                "success": len(expanded_path) > 0,
                "reward": reward
            })
            results.append({
                "user_id": u,
                "success": len(expanded_path) > 0,
                "reward": reward
            })

    df_data_rates = pd.DataFrame(all_user_data_rates, columns=["user_id", "time", "sat", "channel", "data_rate"])
    results_df = pd.DataFrame(results)
    return results_df, all_user_paths, load_by_time, df_data_rates

