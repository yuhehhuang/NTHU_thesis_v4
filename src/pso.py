# -*- coding: utf-8 -*-
# src/pso.py
# 粒子群演算法 (PSO) 版本的批次分配器：
# - 逐時間 t 處理 t_start==t 的使用者（batch）
# - 用 PSO 搜尋「這批使用者的處理順序」（連續優先度向量 -> 依排序得到順序）
# - 依該順序做可行性檢查與“貪婪指派”（挑該 user 目前最佳的 (sat, ch)）
# - 指派時遵守：W-slot 可視、busy-until 容量、同頻干擾（用 snapshot 傳給 compute_sinr_and_rate）
#
# 回傳：
#   df_results: user_id,time,sat,channel,data_rate
#   df_paths  : user_id,path(str(list[(sat,ch,t)])),t_begin,t_end,success,reward
#   load_by_time: dict[t][sat] = used_ch_count

from __future__ import annotations
from collections import defaultdict
import ast
import math
import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from time import perf_counter
from src.utils import compute_sinr_and_rate
import os

# ---------- 小工具：busy-until -> snapshot(dict[ sat ][ ch ] = 0/1) ----------
def _build_snapshot(busy_until: Dict[str, Dict[int, int]], t_now: int) -> Dict[str, Dict[int, int]]:
    return {sat: {ch: 1 if until >= t_now else 0 for ch, until in chs.items()}
            for sat, chs in busy_until.items()}

def _sat_load_ratio(busy_until: Dict[str, Dict[int, int]], sat: str, t_now: int) -> float:
    used = sum(1 for _ch, until in busy_until[sat].items() if until >= t_now)
    tot  = len(busy_until[sat])
    return (used / tot) if tot > 0 else 0.0

def _visible_sats_at(df_access: pd.DataFrame, t: int) -> List[str]:
    row = df_access[df_access["time_slot"] == t]
    if row.empty: return []
    vs = row["visible_sats"].iloc[0]
    return ast.literal_eval(vs) if isinstance(vs, str) else list(vs)

def _future_visible(df_access: pd.DataFrame, sat: str, t: int, t_last: int) -> bool:
    for ft in range(t, t_last + 1):
        row = df_access[df_access["time_slot"] == ft]
        if row.empty: 
            return False
        vs = row["visible_sats"].iloc[0]
        vs = ast.literal_eval(vs) if isinstance(vs, str) else vs
        if sat not in vs:
            return False
    return True

# ---------- 打分：凸組合 (rate_norm, 1-load) ----------
def _score_pair(params, rate: float, load_ratio: float, alpha: float) -> float:
    # 把 rate 壓到 0~1（單調遞增），避免 rate 量級蓋掉負載
    K = float(params.get("rate_norm_k", 100.0))  # 依你的 Mbps 範圍微調（80~150 常見）
    rate_norm = rate / (rate + K) if rate >= 0 else 0.0
    load_norm = 1.0 - max(0.0, min(1.0, load_ratio))
    if alpha >= 1.0:  # 避免 alpha=1 景觀太平：保留極小破同分器
        eps = 1e-3
        return (1.0 - eps) * load_norm + eps * rate_norm
    return (1.0 - alpha) * rate_norm + alpha * load_norm


# =============== PSO 主體：用「連續優先度向量」決定處理順序 =================
class _PSOOrder:
    def __init__(self, n_vars: int, params: dict, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n = n_vars
        self.pop = int(params.get("pso_particles", 20))
        self.iters = int(params.get("pso_iters", 15))
        self.w = float(params.get("pso_inertia", 0.72))
        self.c1 = float(params.get("pso_c1", 1.25))
        self.c2 = float(params.get("pso_c2", 1.25))
        self.vmax = float(params.get("pso_vmax", 1.5))

        # 位置（優先度）與速度
        self.X = np.random.randn(self.pop, self.n)
        self.V = np.zeros((self.pop, self.n), dtype=float)

        self.pbest_X = self.X.copy()
        self.pbest_val = np.full(self.pop, -np.inf)

        self.gbest_X = self.X[0].copy()
        self.gbest_val = -np.inf

    @staticmethod
    def _order_from_priority(pri: np.ndarray) -> np.ndarray:
        # 由大到小（高優先度先被處理）
        return np.argsort(-pri, kind="stable")

    def run(self, eval_func):
        """
        eval_func(order: np.ndarray[int]) -> float
        早停參數（可選）請掛在 self 上：
        - self.patience:           int   連續幾代沒有明顯進步就停（0 表示不用）
        - self.tol:                float 進步門檻，相對增益：(new-old)/(|old|+1e-9)
        - self.time_budget_sec:    float 牆鐘時間上限（秒；0 表示不用）
        - self.target_score:       float 達到此分數即停（None 表示不用）
        - self.stop_file:          str   外部停止旗標檔路徑（例如 "STOP.txt"；None 表示不用）
        - self.verbose:            bool  是否列印早停訊息
        """

        # ---- 初始化評分（與你原本相同）----
        for i in range(self.pop):
            val = eval_func(self._order_from_priority(self.X[i]))
            self.pbest_val[i] = val
            self.pbest_X[i] = self.X[i].copy()
            if val > self.gbest_val:
                self.gbest_val = val
                self.gbest_X = self.X[i].copy()

        # ---- 早停設定（預設為關閉）----
        patience        = int(getattr(self, "patience", 0))
        tol             = float(getattr(self, "tol", 0.0))
        time_budget_sec = float(getattr(self, "time_budget_sec", 0.0))
        target_score    = getattr(self, "target_score", None)
        stop_file       = getattr(self, "stop_file", None)
        verbose         = bool(getattr(self, "verbose", False))

        start_time = perf_counter()
        stall = 0
        best_prev = self.gbest_val

        # ---- PSO 主迭代（在每一代尾端檢查早停）----
        for it in range(self.iters):
            r1 = np.random.rand(self.pop, self.n)
            r2 = np.random.rand(self.pop, self.n)
            self.V = self.w * self.V + self.c1 * r1 * (self.pbest_X - self.X) + self.c2 * r2 * (self.gbest_X - self.X)
            self.V = np.clip(self.V, -self.vmax, self.vmax)  # 限速
            self.X = self.X + self.V

            for i in range(self.pop):
                val = eval_func(self._order_from_priority(self.X[i]))
                if val > self.pbest_val[i]:
                    self.pbest_val[i] = val
                    self.pbest_X[i] = self.X[i].copy()
                    if val > self.gbest_val:
                        self.gbest_val = val
                        self.gbest_X = self.X[i].copy()

            # ---- 早停條件 ----
            # 時間上限
            if time_budget_sec and (perf_counter() - start_time) >= time_budget_sec:
                if verbose:
                    print(f"[PSO] stop by time budget at iter={it}, best={self.gbest_val:.6f}")
                break

            # 外部旗標檔
            if stop_file and os.path.exists(stop_file):
                if verbose:
                    print(f"[PSO] stop by flag file '{stop_file}' at iter={it}, best={self.gbest_val:.6f}")
                break

            # 達標分數
            if (target_score is not None) and (self.gbest_val >= float(target_score)):
                if verbose:
                    print(f"[PSO] stop by target score at iter={it}, best={self.gbest_val:.6f}")
                break

            # 無明顯進步（相對增益）
            if tol > 0:
                rel_gain = (self.gbest_val - best_prev) / (abs(best_prev) + 1e-9)
                if rel_gain <= tol:
                    stall += 1
                else:
                    stall = 0
                    best_prev = self.gbest_val
                if patience and stall >= patience:
                    if verbose:
                        print(f"[PSO] stop by patience at iter={it}, best={self.gbest_val:.6f}")
                    break

        return self._order_from_priority(self.gbest_X), self.gbest_val



# =============== 用某個順序做一次「可行的貪婪指派」評分 =================
def _simulate_assign_for_order(
    order_users: List[int],
    t: int,
    df_users: pd.DataFrame,
    df_access: pd.DataFrame,
    path_loss,
    busy_until: Dict[str, Dict[int, int]],
    user_assignments: Dict[int, List[Tuple[int, str, int]]],  # 現有指派（查 last_used_sat）
    params: dict,
    W: int,
    alpha: float,
    commit: bool = False
):
    """
    依 order_users 的順序做一次指派。
    若 commit=False：在 busy_until 的拷貝上模擬，只回傳目標值與方案；
    若 commit=True：真的把結果寫回 (busy_until, user_assignments) 並回傳同時產生的紀錄。
    """
    # 工作副本（模擬）
    local_busy = {sat: chs.copy() for sat, chs in busy_until.items()}

    snapshot_t = _build_snapshot(local_busy, t)
    total_score = 0.0
    records = []   # for results.csv
    chosen_pairs = {}  # uid -> (sat, ch, t_last, data_rate)

    for uid in order_users:
        # user 還沒到 t 或已結束就跳過（通常外面會過濾好）
        t_end = int(df_users.loc[uid, "t_end"])
        if t > t_end:
            continue

        # 這個 user 在上一刻用的是哪顆衛星？
        last_used_sat = user_assignments[uid][-1][1] if user_assignments.get(uid) else None

        visible_sats = _visible_sats_at(df_access, t)
        best = None  # (score, sat, ch, t_last, data_rate)

        for sat in visible_sats:
            if sat not in local_busy:
                continue
            if (sat, t) not in path_loss:
                continue
            for ch, until in local_busy[sat].items():
                if until >= t:
                    continue  # 這個 channel 在 t 仍被佔用

                # 決定要固定到哪一刻
                will_handover = (last_used_sat != sat)
                t_last = min(t + W - 1, t_end) if will_handover else t

                # 檢查未來 W-slot 可視 & 通道空閒
                feasible = True
                if not _future_visible(df_access, sat, t, t_last):
                    feasible = False
                else:
                    for ft in range(t, t_last + 1):
                        if local_busy[sat][ch] >= ft:
                            feasible = False
                            break
                if not feasible:
                    continue

                # 用“目前 snapshot_t”計算 rate（含 co-channel）
                try:
                    _, rate = compute_sinr_and_rate(params, path_loss, sat, t, snapshot_t, ch)
                except Exception:
                    rate = 0.0

                load_ratio = _sat_load_ratio(local_busy, sat, t)
                score = _score_pair(params, rate, load_ratio, alpha)

                if (best is None) or (score > best[0]):
                    best = (score, sat, ch, t_last, rate)

        if best is None:
            continue  # 這個 user 此刻配不到

        score, sat, ch, t_last, rate = best
        total_score += score
        chosen_pairs[uid] = (sat, ch, t_last, rate)

        # 更新 local 狀態（供後續 user 評分與可行性用）
        for ft in range(t, t_last + 1):
            local_busy[sat][ch] = max(local_busy[sat][ch], t_last)
        # 立刻更新 snapshot_t 的佔用（讓下一個 user 看見這次選擇造成的干擾）
        snapshot_t[sat][ch] = 1

        # 暫記 records（真正 commit 才展開成逐時刻）
        if commit:
            # 寫回真正狀態
            busy_until[sat][ch] = max(busy_until[sat][ch], t_last)
            # 寫回 user 路徑與當刻的資料率（注意：資料率只算 t 當刻）
            for used_t in range(t, t_last + 1):
                # 保存逐時刻紀錄（data_rate 用當刻的 rate）
                records.append((uid, used_t, sat, ch, rate))

            # 更新 user_assignments（和你原本格式一致）
            if uid not in user_assignments:
                user_assignments[uid] = []
            for used_t in range(t, t_last + 1):
                user_assignments[uid].append((used_t, sat, ch))

    return total_score, chosen_pairs, records


# ============================ 對外主函式 ==============================
def run_pso_per_W(df_users, df_access, path_loss, sat_channel_dict_backup, sat_positions, params, W):
    """
    PSO 版：逐 t 批次分配（t_start==t 的使用者），
    粒子群搜尋這批 user 的“處理順序”，再依順序做可行的貪婪指派。
    """
    time_slots = int(df_access["time_slot"].max()) + 1 if "time_slot" in df_access.columns else len(df_access)
    alpha = float(params.get("alpha", 0.0))
    seed = params.get("seed", None)

    # busy-until：所有通道初始化為 -1（代表已釋放）
    busy_until = {sat: {ch: -1 for ch in chs.keys()} for sat, chs in sat_channel_dict_backup.items()}

    user_assignments: Dict[int, List[Tuple[int, str, int]]] = defaultdict(list)
    data_rate_records: List[Tuple[int,int,str,int,float]] = []
    load_by_time = defaultdict(lambda: defaultdict(int))

    # 這兩個 dict 用來管理使用者在批次迴圈中的狀態
    next_available_time = {uid: int(df_users.loc[uid, "t_start"]) for uid in df_users.index}
    all_users = set(df_users.index)
    remaining_users = set(df_users.index)

    t = 0
    rng = random.Random(seed)
    while t < time_slots:
        # 釋放邏輯由 busy-until 自然處理，這裡只抓「此刻可參賽的 user」
        candidate_users = [uid for uid in remaining_users
                           if next_available_time[uid] == t and int(df_users.loc[uid, "t_end"]) >= t]
        if not candidate_users:
            t += 1
            continue

        # 可選：把批次切成多段（降低尖峰；預設 1 段）
        num_splits = max(1, int(params.get("pso_stage_splits", 1)))
        # 均分為 num_splits 段
        chunks = np.array_split(candidate_users, num_splits)

        for chunk in chunks:
            chunk = list(chunk)
            if not chunk:
                continue

            # --- 定義 PSO 評分：給定一個 user 順序 → 模擬一次分派，回總分 ---
            def _eval(order_idx: np.ndarray) -> float:
                order_users = [chunk[i] for i in order_idx]
                total, _, _ = _simulate_assign_for_order(
                    order_users, t, df_users, df_access, path_loss,
                    busy_until, user_assignments, params, W, alpha, commit=False
                )
                return total

            # --- 啟動 PSO 搜索順序 ---
            pso = _PSOOrder(n_vars=len(chunk), params=params, seed=seed)
            # 初始解：加入一個啟發式（避免太隨機）：以「各 user 當下最佳 score」排序
            # 這個啟發式透過手動覆蓋一個粒子的 X 來注入
            heur_pri = []
            snapshot_t = _build_snapshot(busy_until, t)
            for uid in chunk:
                last_used_sat = user_assignments[uid][-1][1] if user_assignments.get(uid) else None
                vs = _visible_sats_at(df_access, t)
                best_sc = -1e9
                for sat in vs:
                    if sat not in busy_until or (sat, t) not in path_loss:
                        continue
                    for ch, until in busy_until[sat].items():
                        if until >= t: 
                            continue
                        will_handover = (last_used_sat != sat)
                        t_end = int(df_users.loc[uid, "t_end"])
                        t_last = min(t + W - 1, t_end) if will_handover else t
                        # 可視+空閒檢查
                        feasible = True
                        if not _future_visible(df_access, sat, t, t_last):
                            feasible = False
                        else:
                            for ft in range(t, t_last + 1):
                                if busy_until[sat][ch] >= ft:
                                    feasible = False
                                    break
                        if not feasible:
                            continue
                        try:
                            _, rate = compute_sinr_and_rate(params, path_loss, sat, t, snapshot_t, ch)
                        except Exception:
                            rate = 0.0
                        load_ratio = _sat_load_ratio(busy_until, sat, t)
                        sc = _score_pair(params, rate, load_ratio, alpha)
                        if sc > best_sc: best_sc = sc
                heur_pri.append(best_sc if best_sc > -1e8 else -1e6)
            if len(heur_pri) == len(chunk):
                # 讓第 0 號粒子用啟發式優先度
                pso.X[0, :] = np.array(heur_pri, dtype=float)

            best_order_idx, _ = pso.run(_eval)
            best_order_users = [chunk[i] for i in best_order_idx]

            # --- 用最佳順序真正 commit ---
            _, chosen, recs = _simulate_assign_for_order(
                best_order_users, t, df_users, df_access, path_loss,
                busy_until, user_assignments, params, W, alpha, commit=True
            )
            data_rate_records.extend(recs)

            # 更新 next_available_time / remaining_users
            for uid in best_order_users:
                if uid not in chosen:
                    # 沒配到，下回合再來
                    next_available_time[uid] = t + 1
                else:
                    _sat, _ch, t_last, _rate = chosen[uid]
                    next_available_time[uid] = t_last + 1
                    if next_available_time[uid] > int(df_users.loc[uid, "t_end"]):
                        if uid in remaining_users:
                            remaining_users.remove(uid)

            # 更新 load_by_time（只需要對 t..t_last 累積）
            for uid, (_sat, _ch, t_last, _rate) in chosen.items():
                for used_t in range(t, t_last + 1):
                    load_by_time[used_t][_sat] += 1

        # 下一個時間
        t += 1

    # === 輸出 DataFrame ===
    df_results = pd.DataFrame(data_rate_records, columns=["user_id", "time", "sat", "channel", "data_rate"])

    # 轉換成 path 格式（與 greedy 相同）
    formatted_paths = []
    for uid, entries in user_assignments.items():
        if not entries: 
            continue
        entries = sorted(entries, key=lambda x: x[0])
        path_list = [(sat, ch, tt) for (tt, sat, ch) in entries]
        t_begin = entries[0][0]
        t_end = entries[-1][0]
        success = (t_end - t_begin + 1) == (int(df_users.loc[uid, "t_end"]) - int(df_users.loc[uid, "t_start"]) + 1)
        # 把該 user 的 data_rate 相加（以 df_results 節錄）
        if not df_results.empty:
            total_rate = float(df_results.loc[df_results["user_id"] == uid, "data_rate"].sum())
        else:
            total_rate = 0.0
        formatted_paths.append([uid, str(path_list), t_begin, t_end, success, total_rate])

    df_paths = pd.DataFrame(formatted_paths, columns=["user_id", "path", "t_begin", "t_end", "success", "reward"])
    return df_results, df_paths, load_by_time
