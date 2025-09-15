# ga.py
#
#elif METHOD == "ga":
#    ga = GeneticAlgorithm(
#        population_size=10,
#        user_df=df_users,
#        access_matrix=df_access.to_dict(orient="records"),
#        W=W,
#        path_loss=path_loss,
#        sat_channel_dict=copy.deepcopy(sat_channel_dict_backup),
#        params=params,
#        seed=123456  # ✅ 固定一個整數 seed；不想固定就拿掉這行
#    )
#    ga.evolve(generations=5)  # 訓練 5 輪(5輪大概要1小時)，可調整為 20、50 等等
#    results_df, all_user_paths, load_by_time, df_data_rates = ga.export_best_result()
# -*- coding: utf-8 -*-
"""
Genetic Algorithm (GA) for user path allocation with W-slot handover constraint.

本檔案包含：
1) Individual：代表一組完整的解（所有 user 的 path 分配）
   - 使用 greedy 方式產生初始路径（generate_fast_path）
   - 提供 rebuild_from_position() 依既定 path 重建 reward 與 data_rates
2) GeneticAlgorithm：演化流程（選擇、交配、突變）
   - crossover 時以使用者為單位從雙親拷貝 path
   - mutate 時以 DP 為單位對單一 user 嘗試重新規劃 path

注意：
- 本版本僅新增中文註解與小幅排版，不改動既有邏輯。
"""

import random
import copy
import pandas as pd
from collections import defaultdict

from src.dp import run_dp_path_for_user
from src.utils import (
    compute_sinr_and_rate,
    compute_score,
    update_m_s_t_from_channels,
    check_visibility,
)


class Individual:
    """
    一個個體（解）：表示所有 user 的 path 分配方案。
    內含：
      - position: dict[user_id] -> List[(sat, ch, t)]
      - data_rates: List[(user_id, t, sat, ch, data_rate)]
      - reward: float，整體 reward（依 compute_score 累積）

    參數：
      user_df: DataFrame，至少包含 user_id, t_start, t_end 欄位
      access_matrix: dict[int -> {"visible_sats": set/iterable}]，每個時間能見衛星
      W: int，每 W 個 time slot 才允許換手（首換手不受限）
      path_loss: 你的路損/通道模型資料
      sat_channel_dict: dict[sat][ch] = 0/1/…，表示該衛星各 channel 的占用（背景&已分配）
      params: 其他各式計算參數（compute_* 會用到）
      seed: 隨機種子，影響 greedy 內部的隨機順序
    """

    def __init__(self, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.user_df = user_df
        self.access_matrix = access_matrix
        self.df_access = pd.DataFrame(access_matrix)  # check_visibility 會用到的 DataFrame 版本
        self.W = W
        self.path_loss = path_loss
        self.sat_channel_dict = sat_channel_dict
        self.params = params
        self.rng = random.Random(seed)

        # 解的核心資料結構
        self.position = {}      # 各 user 的 path
        self.data_rates = []    # 所有 (user_id, t, sat, ch, data_rate)
        self.reward = 0.0       # 總 reward

        # 以 greedy 初始化（同一 t_start 批次內打亂 user 順序）
        self.generate_fast_path()

    # -----------------------------
    # Greedy 產生初始解（建議作為 GA 的個體初始化）
    # -----------------------------
    def generate_fast_path(self):
        # 重置解的內容
        self.position = {}
        self.data_rates = []
        self.reward = 0.0

        # 使用 tmp_sat_dict 追蹤「當前」分配階段的 (sat, ch) 佔用狀態
        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)
        total_reward = 0.0

        # 追蹤仍在使用資源的 user（用於在下一批 t_start 前釋放已結束者）
        active_user_paths = []

        # 依 t_start 升冪處理，每個 t_start 批次內把使用者順序打亂
        for t_val, group_df in self.user_df.sort_values("t_start").groupby("t_start"):
            users = list(group_df.itertuples(index=False))
            self.rng.shuffle(users)

            for user in users:
                user_id = int(user.user_id)
                t_start = int(user.t_start)
                t_end = int(user.t_end)

                # === 在此批 t_start 前，釋放已完成的使用者資源 ===
                to_remove = []
                for old_user in active_user_paths:
                    if old_user["t_end"] < t_start:
                        # 以 (s, c) pair 去重，避免同 sat/ch 在同一 user path 被重複釋放
                        for s, c in set((s, c) for s, c, _ in old_user["path"]):
                            tmp_sat_dict[s][c] = max(0, tmp_sat_dict[s][c] - 1)
                        to_remove.append(old_user)
                for u in to_remove:
                    active_user_paths.remove(u)

                # === 該 user 的 path 建構變數 ===
                t = t_start
                current_sat, current_ch = None, None
                last_ho_time = t_start         # 最近一次換手時間
                is_first_handover = True       # 是否為「第一次換手」（不受 W 限制）

                user_path = []                 # 此 user 的完整 path: [(sat, ch, t), ...]
                data_rate_records = []         # 此 user 的 data_rate 記錄
                user_reward = 0.0              # 此 user reward 累積

                # -------------------------------------------------
                # 1) 初次選擇：在 t_start 選一個可用 sat/ch （未被占用、可見、data_rate > 0）
                # -------------------------------------------------
                best_sat, best_ch, best_score, best_data_rate = None, None, float("-inf"), 0.0

                for sat in self.access_matrix[t_start]["visible_sats"]:
                    for ch in tmp_sat_dict[sat]:
                        if tmp_sat_dict[sat][ch] > 0:
                            continue  # 該 channel 目前已被占用
                        _, data_rate = compute_sinr_and_rate(self.params, self.path_loss, sat, t_start, tmp_sat_dict, ch)
                        if data_rate is None or data_rate <= 0:
                            continue
                        # m_s_t 代表每顆衛星在「當下」的負載（由 tmp_sat_dict 推導）
                        m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                        score = compute_score(self.params, m_s_t, data_rate, sat)

                        # 加上極小隨機項，避免完全相同分數導致不可預期的 tie
                        score += 1e-9 * self.rng.random()

                        if score > best_score:
                            best_score = score
                            best_sat, best_ch = sat, ch
                            best_data_rate = data_rate

                # 若找不到任何可用 sat/ch，表示此 user 無法服務，空 path
                if best_sat is None:
                    self.position[user_id] = []
                    continue

                # 套用初次選擇
                current_sat, current_ch = best_sat, best_ch
                user_path.append((current_sat, current_ch, t_start))
                data_rate_records.append((user_id, t_start, current_sat, current_ch, best_data_rate))
                m_s_t0 = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                user_reward += compute_score(self.params, m_s_t0, best_data_rate, current_sat)

                # -------------------------------------------------
                # 2) 後續時間：每一步檢查是否允許換手（W-slot 規則）
                #    若換手，則「跳 W 格」；否則每次前進 1 個 slot
                # -------------------------------------------------
                t = t_start + 1
                while t <= t_end:
                    can_handover = is_first_handover or (t - last_ho_time >= self.W)
                    did_handover = False

                    # 預設不換手：候選依然為 current
                    best_sat, best_ch, best_score = current_sat, current_ch, float("-inf")

                    if can_handover:
                        # 候選衛星、候選通道順序打亂，讓不同 seed 能探索更多組合
                        vsats = list(self.access_matrix[t]["visible_sats"])
                        self.rng.shuffle(vsats)

                        for sat in vsats:
                            ch_list = list(tmp_sat_dict[sat].keys())
                            self.rng.shuffle(ch_list)

                            for ch in ch_list:
                                if tmp_sat_dict[sat][ch] > 0:
                                    continue  # 通道占用
                                # W-slot 可視性檢查：至少要能撐到下一次允許換手前
                                if not check_visibility(self.df_access, sat, t, min(t_end, t + self.W - 1)):
                                    continue
                                _, dr0 = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                                if dr0 is None or dr0 <= 0:
                                    continue
                                m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                                score0 = compute_score(self.params, m_s_t, dr0, sat) + 1e-9 * self.rng.random()
                                if score0 > best_score:
                                    best_score = score0
                                    best_sat, best_ch = sat, ch

                        # 若挑到不同的 (sat, ch)，表示發生換手
                        if (best_sat is not None) and (best_sat != current_sat or best_ch != current_ch):
                            current_sat, current_ch = best_sat, best_ch
                            last_ho_time = t
                            is_first_handover = False
                            did_handover = True

                    # 換手成功 => step = W（代表這段期間鎖定新連線）
                    # 否則 => step = 1（持續原連線逐格前進）
                    step = self.W if did_handover else 1

                    for w in range(step):
                        tt = t + w
                        if tt > t_end:
                            break
                        # 若未換手且當前衛星在 tt 不可見，就中止此段追蹤
                        if not did_handover:
                            if current_sat not in self.access_matrix[tt]["visible_sats"]:
                                break

                        _, dr = compute_sinr_and_rate(
                            self.params, self.path_loss, current_sat, tt, tmp_sat_dict, current_ch
                        )

                        user_path.append((current_sat, current_ch, tt))
                        data_rate_records.append((user_id, tt, current_sat, current_ch, dr if dr else 0.0))

                        if dr and dr > 0:
                            m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                            user_reward += compute_score(self.params, m_s_t, dr, current_sat)

                    t += step

                # === 收尾：若 path 非空，登記資源占用、累積 reward 與資料率 ===
                if user_path:
                    self.position[user_id] = user_path
                    # 以 (sat, ch) pair 去重後再 +1，避免同一 user 在多個 t 重複加總
                    for s, c in set((s, c) for s, c, _ in user_path):
                        tmp_sat_dict[s][c] += 1

                    self.data_rates.extend(data_rate_records)
                    total_reward += user_reward

                    # 放入 active pool，之後等到它 t_end 之後會釋放資源
                    active_user_paths.append({
                        "user_id": user_id,
                        "path": user_path,
                        "t_end": t_end
                    })
                else:
                    # 無法服務（找不到初始連線或中途全失敗）
                    self.position[user_id] = []

        self.reward = total_reward

    # -------------------------------------
    # 單一 user 的「純 greedy」建路（供測試/比較用）
    # -------------------------------------
    def _run_greedy_path(self, user_id, t_start, t_end, tmp_sat_dict):
        """
        僅對單一 user 以 greedy 方式建 path（不會影響 self.position）。
        回傳: (user_path, data_rate_records, user_reward)
        """
        user_path = []
        data_rate_records = []
        user_reward = 0.0

        t = t_start
        current_sat, current_ch = None, None
        last_ho_time = t_start
        is_first_handover = True

        # === 第一次選擇 ===
        best_sat, best_ch, best_score, best_dr = None, None, float("-inf"), 0.0
        for sat in self.access_matrix[t_start]["visible_sats"]:
            for ch in tmp_sat_dict[sat]:
                if tmp_sat_dict[sat][ch] > 0:
                    continue
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t_start, tmp_sat_dict, ch)
                if dr and dr > 0:
                    score = compute_score(
                        self.params,
                        update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()),
                        dr,
                        sat
                    )
                    score += 1e-9 * self.rng.random()
                    if score > best_score:
                        best_score = score
                        best_sat, best_ch, best_dr = sat, ch, dr

        if best_sat is None:
            return [], [], 0.0

        current_sat, current_ch = best_sat, best_ch
        user_path.append((current_sat, current_ch, t_start))
        data_rate_records.append((user_id, t_start, current_sat, current_ch, best_dr))
        user_reward += best_score

        # === 後續時間 ===
        t += 1
        while t <= t_end:
            can_ho = is_first_handover or (t - last_ho_time >= self.W)
            did_ho = False
            best_sat, best_ch, best_score = current_sat, current_ch, float("-inf")

            if can_ho:
                vsats = list(self.access_matrix[t]["visible_sats"])
                self.rng.shuffle(vsats)
                for sat in vsats:
                    chs = list(tmp_sat_dict[sat].keys())
                    self.rng.shuffle(chs)
                    for ch in chs:
                        if tmp_sat_dict[sat][ch] > 0:
                            continue
                        if not check_visibility(self.df_access, sat, t, min(t_end, t + self.W - 1)):
                            continue
                        _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                        if dr and dr > 0:
                            score = compute_score(
                                self.params,
                                update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()),
                                dr,
                                sat
                            )
                            score += 1e-9 * self.rng.random()
                            if score > best_score:
                                best_score = score
                                best_sat, best_ch = sat, ch

                if (best_sat != current_sat) or (best_ch != current_ch):
                    current_sat, current_ch = best_sat, best_ch
                    last_ho_time = t
                    is_first_handover = False
                    did_ho = True

            step = self.W if did_ho else 1
            for w in range(step):
                tt = t + w
                if tt > t_end:
                    break
                if not did_ho and current_sat not in self.access_matrix[tt]["visible_sats"]:
                    break
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, current_sat, tt, tmp_sat_dict, current_ch)
                user_path.append((current_sat, current_ch, tt))
                data_rate_records.append((user_id, tt, current_sat, current_ch, dr if dr else 0.0))
                if dr and dr > 0:
                    user_reward += compute_score(
                        self.params,
                        update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()),
                        dr,
                        current_sat
                    )
            t += step

        return user_path, data_rate_records, user_reward

    # ---------------------------------------------------------
    # 依照既有 self.position 重建 data_rates 與 reward（離線評分）
    # ---------------------------------------------------------
    def rebuild_from_position(self):
        """
        依 self.position（既定 path）重算：
          - 每個 time 的 data_rate
          - 總 reward
        不會改動 self.position，但會根據使用到的 (sat, ch) 更新臨時載入。
        """
        self.data_rates = []
        self.reward = 0.0

        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)
        total_reward = 0.0

        # 以 t_start 升冪處理，並維持 active pool 來正確釋放過期資源
        df_ts = self.user_df[["user_id", "t_start", "t_end"]].copy()
        df_ts = df_ts.sort_values("t_start")

        active_user_paths = []

        for _, row in df_ts.iterrows():
            user_id = int(row["user_id"])
            t_start = int(row["t_start"])
            t_end = int(row["t_end"])

            # 在此 t_start 前釋放已完成之 user 的 (sat, ch) 占用
            to_remove = []
            for old_user in active_user_paths:
                if old_user["t_end"] < t_start:
                    for s, c in set((s, c) for s, c, _ in old_user["path"]):
                        tmp_sat_dict[s][c] = max(0, tmp_sat_dict[s][c] - 1)
                    to_remove.append(old_user)
            for u in to_remove:
                active_user_paths.remove(u)

            path = self.position.get(user_id, [])
            if not path:
                continue

            # 依時間排序，確保計分的時間順序正確
            path = sorted(path, key=lambda x: x[2])

            user_reward = 0.0
            used_pairs = set()

            for sat, ch, t in path:
                # 若該時刻不可見，略過本 slot 的計分（但 path 仍保留）
                if sat not in self.access_matrix[t]["visible_sats"]:
                    continue

                _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)

                # 紀錄 data_rate（即使 dr 為 None 或 <=0，也做 0.0 記錄）
                self.data_rates.append((user_id, t, sat, ch, dr if dr else 0.0))

                # 只有正 data_rate 才會貢獻 reward
                if dr and dr > 0:
                    m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                    user_reward += compute_score(self.params, m_s_t, dr, sat)
                    used_pairs.add((sat, ch))

            # 此 user 實際使用到的 (sat, ch) 於 tmp_sat_dict 上 +1（避免同一 user 重複 +1）
            for s, c in used_pairs:
                tmp_sat_dict[s][c] = tmp_sat_dict[s].get(c, 0) + 1

            total_reward += user_reward

            # 放進 active pool，待其 t_end 之後釋放
            active_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_end": t_end
            })

        self.reward = total_reward


class GeneticAlgorithm:
    """
    GA 主流程：
      - 初始化：以多個不同 seed 的 greedy 個體組成族群
      - evolve：每代保留 elite，透過 tournament 選雙親、crossover、mutation 形成新世代
      - crossover：以 user 為粒度，隨機從 p2 拷貝該 user 的 path
      - mutate：對部分 user 以 DP 嘗試改善 path，成功則更新
      - rebuild_from_position：在評分前重建 data_rates / reward

    參數：
      population_size: 族群大小
      user_df, access_matrix, W, path_loss, sat_channel_dict, params: 與 Individual 相同
      seed: 整體隨機種子，會衍生出每個個體的種子
    """

    def __init__(self, population_size, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.population_size = population_size
        self.user_df = user_df
        self.access_matrix = access_matrix
        self.W = W
        self.path_loss = path_loss
        self.params = params
        self.seed_base = seed or random.randint(0, 999_999)

        # 以不同 seed 產生初始族群（每個 Individual 內部 greedy 會有不同隨機性）
        self.population = [
            Individual(
                user_df, access_matrix, W, path_loss, copy.deepcopy(sat_channel_dict), params,
                seed=self.seed_base + i * 7919  # 使用質數步幅，降低種子碰撞
            )
            for i in range(population_size)
        ]
        # 依 reward 由大到小排序
        self.population.sort(key=lambda x: x.reward, reverse=True)
        # 保存目前最佳個體
        self.best_individual = copy.deepcopy(self.population[0])

    def evolve(self, generations, elite_size=2, mutation_rate=0.2):
        """
        執行多代演化。
        - elite_size: 每代直接保留的菁英個體數
        - mutation_rate: 每位 user 觸發 DP 突變的機率
        """
        for gen in range(generations):
            # 1) 菁英保留
            next_gen = self.population[:elite_size]

            # 2) 產生後代直到湊滿族群大小
            while len(next_gen) < self.population_size:
                p1, p2 = self.tournament_selection(), self.tournament_selection()
                child = self.crossover(p1, p2)
                self.mutate(child, mutation_rate)
                next_gen.append(child)

            # 3) 評分：重建 data_rates / reward
            for ind in next_gen:
                ind.rebuild_from_position()

            # 4) 新族群就位、更新最佳解
            self.population = sorted(next_gen, key=lambda x: x.reward, reverse=True)
            if self.population[0].reward > self.best_individual.reward:
                self.best_individual = copy.deepcopy(self.population[0])

    def tournament_selection(self, k=3):
        """
        錦標賽選擇：隨機抽 k 個，取 reward 最高者。
        """
        return max(random.sample(self.population, k), key=lambda x: x.reward)

    def crossover(self, p1, p2):
        """
        交配：以 p1 為骨幹，對每個 user 以 50% 機率用 p2 的 path 覆蓋。
        交配後先重建一次（讓 child 有正確的 reward / data_rates）。
        """
        child = copy.deepcopy(p1)
        for uid in child.position:
            if random.random() < 0.5:
                child.position[uid] = copy.deepcopy(p2.position.get(uid, []))
        child.rebuild_from_position()
        return child

    def mutate(self, individual, mutation_rate):
        """
        突變：逐 user 以 mutation_rate 機率觸發 DP 重新規劃該 user path。
        - 成功(success=True)才會更新 position[uid]
        - 至少有一位 user 被更新後，最後再重建一次
        """
        mutated = False
        for user in individual.user_df.itertuples():
            if random.random() < mutation_rate:
                uid = int(user.user_id)
                t_start, t_end = int(user.t_start), int(user.t_end)

                # 建構 DP 的快照：把其它 user 已用的 (sat, ch) 累計進去
                snapshot = self._build_snapshot(individual, uid)

                # 使用 DP 求單一 user 在目前快照下的最佳路徑
                path, reward, success, data_rates = run_dp_path_for_user(
                    uid, t_start, t_end, self.W, self.access_matrix,
                    self.path_loss, snapshot, self.params
                )

                if success:
                    individual.position[uid] = path
                    mutated = True

        if mutated:
            individual.rebuild_from_position()

    def _build_snapshot(self, individual, exclude_user_id):
        """
        以個體 current position 生成 (sat, ch) 佔用快照，但不包含 exclude_user_id 自己。
        用於 DP 嘗試替該 user 規劃時的環境假設。
        """
        tmp = copy.deepcopy(individual.sat_channel_dict)
        for uid, path in individual.position.items():
            if uid == exclude_user_id:
                continue
            for s, c, _ in path:
                tmp[s][c] += 1
        return tmp

    def export_best_result(self):
        """
        匯出最佳個體的：
          - results: DataFrame，逐 user 的成功/失敗
          - all_user_paths: list[{user_id, path, t_begin, t_end, success, reward(None)}]
          - load_by_time: dict[time][sat] = 使用中的 channel 數
          - df_data_rates: DataFrame(user_id, time, sat, channel, data_rate)

        注意：reward 欄位此處填 None（如需可再計）。
        """
        best = self.best_individual
        all_user_paths = []
        load_by_time = defaultdict(lambda: defaultdict(int))
        results = []

        for user_id, path in best.position.items():
            if not path:
                results.append({"user_id": user_id, "reward": None, "success": False})
                continue

            t_begin = min(t for _, _, t in path)
            t_end = max(t for _, _, t in path)

            # 累加每個時間、每顆衛星的使用量（不分 channel 類型，單純 +1）
            for s, c, t in path:
                load_by_time[t][s] += 1

            all_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_begin": t_begin,
                "t_end": t_end,
                "success": True,
                "reward": None
            })
            results.append({"user_id": user_id, "reward": None, "success": True})

        df_data_rates = pd.DataFrame(best.data_rates, columns=["user_id", "time", "sat", "channel", "data_rate"])

        return pd.DataFrame(results), all_user_paths, load_by_time, df_data_rates
