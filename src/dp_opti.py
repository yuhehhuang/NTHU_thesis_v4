import pandas as pd
from collections import defaultdict
import copy
from src.utils import compute_sinr_and_rate, compute_score, update_m_s_t_from_channels
from src.dp import run_dp_path_for_user  # 沿用原本的 DP 單人路徑尋找

def run_batch_optimal_dp_per_W(
    user_df: pd.DataFrame,
    access_matrix: list,
    path_loss: dict,
    sat_load_dict_backup: dict,
    params: dict,
    W: int = 4
):
    sat_load_dict = {sat: chs.copy() for sat, chs in sat_load_dict_backup.items()}
    user_df = user_df.sort_values(by="t_start").reset_index(drop=True)
    active_user_paths = []
    all_user_paths = []
    results = []
    load_by_time = defaultdict(lambda: defaultdict(int))
    all_user_data_rates = []

    max_time = user_df["t_end"].max()

    for t in range(max_time + 1):
        batch_users = user_df[user_df["t_start"] == t]
        batch_users = batch_users.copy().reset_index(drop=True)
        if batch_users.empty:
            continue

        # 釋放已完成使用者的資源
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t:
                path = old_user.get("path")
                if path:
                    unique_sat_ch = set((s, c) for s, c, _ in path)
                    for sat, ch in unique_sat_ch:
                        sat_load_dict[sat][ch] = max(0, sat_load_dict[sat][ch] - 1)
                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)

        # 動態分配這一批 user
        unassigned = batch_users.to_dict(orient="records")
        while unassigned:
            results_list = []
            for user in unassigned:
                user_id = int(user["user_id"])
                t_start = int(user["t_start"])
                t_end = int(user["t_end"])

                path, reward, success, data_rate_records = run_dp_path_for_user(
                    user_id=user_id,
                    t_start=t_start,
                    t_end=t_end,
                    W=W,
                    access_matrix=access_matrix,
                    path_loss=path_loss,
                    sat_channel_dict=copy.deepcopy(sat_load_dict),
                    params=params
                )

                if not success:
                    continue

                duration = t_end - t_start + 1
                avg_reward = reward / duration

                results_list.append({
                    "user": user,
                    "avg_reward": avg_reward,
                    "reward": reward,
                    "path": path,
                    "data_rate_records": data_rate_records
                })

            if not results_list:
                break  # 剩下的都找不到可行路徑

            # 選 avg reward 最佳者
            best = max(results_list, key=lambda x: x["avg_reward"])
            user = best["user"]
            path = best["path"]
            reward = best["reward"]
            data_rate_records = best["data_rate_records"]

            # 更新系統
            unique_sat_ch = set((s, c) for s, c, _ in path)
            for sat, ch in unique_sat_ch:
                sat_load_dict[sat][ch] += 1
            for s, c, t_slot in path:
                load_by_time[t_slot][s] += 1

            all_user_data_rates.extend(data_rate_records)

            user_id = int(user["user_id"])
            t_start = int(user["t_start"])
            t_end = int(user["t_end"])

            active_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_begin": t_start,
                "t_end": t_end,
                "success": True,
                "reward": reward
            })

            all_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_begin": t_start,
                "t_end": t_end,
                "success": True,
                "reward": reward
            })

            results.append({
                "user_id": user_id,
                "reward": reward,
                "success": True
            })

            # 將該 user 從 unassigned 中移除
            unassigned = [u for u in unassigned if int(u["user_id"]) != user_id]

    df_data_rates = pd.DataFrame(all_user_data_rates, columns=["user_id", "time", "sat", "channel", "data_rate"])
    return pd.DataFrame(results), all_user_paths, load_by_time, df_data_rates
