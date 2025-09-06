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
import random
import copy
import pandas as pd
from collections import defaultdict
from src.dp import run_dp_path_for_user
from src.utils import compute_sinr_and_rate, compute_score, update_m_s_t_from_channels, check_visibility

class Individual:
    """
    一個個體，表示所有 user 的 path 分配方案。
    每個個體有：position（user 的 path）、data_rates、reward
    """
    def __init__(self, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.user_df = user_df
        self.access_matrix = access_matrix
        self.df_access = pd.DataFrame(access_matrix)
        self.W = W
        self.path_loss = path_loss
        self.sat_channel_dict = sat_channel_dict
        self.params = params
        self.rng = random.Random(seed)

        self.position = {}      # user 的 path
        self.data_rates = []    # 所有 user 的 data_rate 記錄
        self.reward = 0         # 總 reward

        self.generate_fast_path()  # 用 greedy 初始化

    def generate_fast_path(self):
        self.position = {}
        self.data_rates = []
        self.reward = 0

        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)
        total_reward = 0

        active_user_paths = []  # ⬅️ 正在使用資源的 user 清單

        # ✅ 依 t_start 遞增排序，再 groupby t_start；每個批次內打亂
        for t_val, group_df in self.user_df.sort_values("t_start").groupby("t_start"):
            users = list(group_df.itertuples(index=False))
            self.rng.shuffle(users)   # 只打亂同一個 t_start 批次內的順序

            for user in users:
                user_id = int(user.user_id)
                t_start = int(user.t_start)
                t_end = int(user.t_end)

                # ✅ 釋放已完成使用者的資源
                to_remove = []
                for old_user in active_user_paths:
                    if old_user["t_end"] < t_start:
                        for s, c in set((s, c) for s, c, _ in old_user["path"]):
                            tmp_sat_dict[s][c] = max(0, tmp_sat_dict[s][c] - 1)
                        to_remove.append(old_user)
                for u in to_remove:
                    active_user_paths.remove(u)

                # 初始化 user path 建構變數
                t = t_start
                current_sat, current_ch = None, None
                last_ho_time = t_start
                is_first_handover = True

                user_path = []
                data_rate_records = []
                user_reward = 0

                # ==========================
                # 1) 第一次選擇（在 t_start）
                # ==========================
                best_sat, best_ch, best_score, best_data_rate = None, None, float("-inf"), 0.0

                for sat in self.access_matrix[t_start]["visible_sats"]:
                    for ch in tmp_sat_dict[sat]:
                        if tmp_sat_dict[sat][ch] > 0:
                            continue
                        _, data_rate = compute_sinr_and_rate(self.params, self.path_loss, sat, t_start, tmp_sat_dict, ch)
                        if data_rate is None or data_rate <= 0:
                            continue
                        m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                        score = compute_score(self.params, m_s_t, data_rate, sat)
                        score += 1e-9 * self.rng.random()
                        if score > best_score:
                            best_score = score
                            best_sat, best_ch = sat, ch
                            best_data_rate = data_rate

                if best_sat is None:
                    self.position[user_id] = []
                    continue

                current_sat, current_ch = best_sat, best_ch
                user_path.append((current_sat, current_ch, t_start))
                data_rate_records.append((user_id, t_start, current_sat, current_ch, best_data_rate))
                m_s_t0 = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                user_reward += compute_score(self.params, m_s_t0, best_data_rate, current_sat)

                # ==========================
                # 2) t_begin 之後
                # ==========================
                t = t_start + 1
                while t <= t_end:
                    can_handover = is_first_handover or (t - last_ho_time >= self.W)
                    did_handover = False

                    best_sat, best_ch, best_score = current_sat, current_ch, float("-inf")

                    if can_handover:
                        vsats = list(self.access_matrix[t]["visible_sats"])
                        self.rng.shuffle(vsats)

                        for sat in vsats:
                            ch_list = list(tmp_sat_dict[sat].keys())
                            self.rng.shuffle(ch_list)
                            for ch in ch_list:
                                if tmp_sat_dict[sat][ch] > 0:
                                    continue
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

                        if (best_sat is not None) and (best_sat != current_sat or best_ch != current_ch):
                            current_sat, current_ch = best_sat, best_ch
                            last_ho_time = t
                            is_first_handover = False
                            did_handover = True

                    step = self.W if did_handover else 1

                    for w in range(step):
                        tt = t + w
                        if tt > t_end:
                            break
                        if not did_handover:
                            if current_sat not in self.access_matrix[tt]["visible_sats"]:
                                break

                        _, dr = compute_sinr_and_rate(self.params, self.path_loss, current_sat, tt, tmp_sat_dict, current_ch)

                        user_path.append((current_sat, current_ch, tt))
                        data_rate_records.append((user_id, tt, current_sat, current_ch, dr if dr else 0.0))

                        if dr and dr > 0:
                            m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                            user_reward += compute_score(self.params, m_s_t, dr, current_sat)

                    t += step

                if user_path:
                    self.position[user_id] = user_path
                    for s, c in set((s, c) for s, c, _ in user_path):
                        tmp_sat_dict[s][c] += 1
                    self.data_rates.extend(data_rate_records)
                    total_reward += user_reward
                    # ✅ 新增：加入活躍 pool
                    active_user_paths.append({
                        "user_id": user_id,
                        "path": user_path,
                        "t_end": t_end
                    })
                else:
                    self.position[user_id] = []

        self.reward = total_reward


    def _run_greedy_path(self, user_id, t_start, t_end, tmp_sat_dict):
        """單獨為一個 user 跑 greedy path"""
        user_path = []
        data_rate_records = []
        user_reward = 0
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
                    score = compute_score(self.params, update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()), dr, sat)
                    score += 1e-9 * self.rng.random()
                    if score > best_score:
                        best_score = score
                        best_sat, best_ch, best_dr = sat, ch, dr

        if best_sat is None:
            return [], [], 0

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
                            score = compute_score(self.params, update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()), dr, sat)
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
                data_rate_records.append((user_id, tt, current_sat, current_ch, dr if dr else 0))
                if dr and dr > 0:
                    user_reward += compute_score(self.params, update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys()), dr, current_sat)
            t += step

        return user_path, data_rate_records, user_reward

    def rebuild_from_position(self):
        self.data_rates = []
        self.reward = 0.0
        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)
        total_reward = 0.0

        df_ts = self.user_df[["user_id", "t_start", "t_end"]].copy()
        df_ts = df_ts.sort_values("t_start")

        active_user_paths = []

        for _, row in df_ts.iterrows():
            user_id = int(row["user_id"])
            t_start = int(row["t_start"])
            t_end = int(row["t_end"])

            # ✅ 釋放已完成使用者的資源
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
            path = sorted(path, key=lambda x: x[2])

            user_reward = 0.0
            used_pairs = set()

            for sat, ch, t in path:
                if sat not in self.access_matrix[t]["visible_sats"]:
                    continue
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                self.data_rates.append((user_id, t, sat, ch, dr if dr else 0.0))
                if dr and dr > 0:
                    m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                    user_reward += compute_score(self.params, m_s_t, dr, sat)
                    used_pairs.add((sat, ch))

            for s, c in used_pairs:
                tmp_sat_dict[s][c] = tmp_sat_dict[s].get(c, 0) + 1

            total_reward += user_reward

            # ✅ 加入 active pool，方便後續釋放
            active_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_end": t_end
            })

        self.reward = total_reward


class GeneticAlgorithm:
    """主要演化流程"""
    def __init__(self, population_size, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.population_size = population_size
        self.user_df = user_df
        self.access_matrix = access_matrix
        self.W = W
        self.path_loss = path_loss
        self.params = params
        self.seed_base = seed or random.randint(0, 999999)

        # 初始化族群
        self.population = [
            Individual(user_df, access_matrix, W, path_loss, copy.deepcopy(sat_channel_dict), params, seed=self.seed_base + i * 7919)
            for i in range(population_size)
        ]
        self.population.sort(key=lambda x: x.reward, reverse=True)
        self.best_individual = copy.deepcopy(self.population[0])

    def evolve(self, generations, elite_size=2, mutation_rate=0.2):
        for gen in range(generations):
            next_gen = self.population[:elite_size]
            while len(next_gen) < self.population_size:
                p1, p2 = self.tournament_selection(), self.tournament_selection()
                child = self.crossover(p1, p2)
                self.mutate(child, mutation_rate)
                next_gen.append(child)

            for ind in next_gen:
                ind.rebuild_from_position()
            self.population = sorted(next_gen, key=lambda x: x.reward, reverse=True)
            if self.population[0].reward > self.best_individual.reward:
                self.best_individual = copy.deepcopy(self.population[0])

    def tournament_selection(self, k=3):
        return max(random.sample(self.population, k), key=lambda x: x.reward)

    def crossover(self, p1, p2):
        child = copy.deepcopy(p1)
        for uid in child.position:
            if random.random() < 0.5:
                child.position[uid] = copy.deepcopy(p2.position.get(uid, []))
        child.rebuild_from_position()
        return child

    def mutate(self, individual, mutation_rate):
        mutated = False
        for user in individual.user_df.itertuples():
            if random.random() < mutation_rate:
                uid = int(user.user_id)
                t_start, t_end = int(user.t_start), int(user.t_end)
                snapshot = self._build_snapshot(individual, uid)
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
        tmp = copy.deepcopy(individual.sat_channel_dict)
        for uid, path in individual.position.items():
            if uid == exclude_user_id:
                continue
            for s, c, _ in path:
                tmp[s][c] += 1
        return tmp

    def export_best_result(self):
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
            for s, c, t in path:
                load_by_time[t][s] += 1
            all_user_paths.append({"user_id": user_id, "path": path, "t_begin": t_begin, "t_end": t_end, "success": True, "reward": None})
            results.append({"user_id": user_id, "reward": None, "success": True})

        df_data_rates = pd.DataFrame(best.data_rates, columns=["user_id", "time", "sat", "channel", "data_rate"])
        return pd.DataFrame(results), all_user_paths, load_by_time, df_data_rates
