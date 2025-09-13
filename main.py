import os
import copy
import pandas as pd
from src.init import load_system_data
from src.greedy import run_greedy_per_W, run_greedy_path_for_user
from src.hungarian import run_hungarian_per_W
from src.utils import recompute_all_data_rates
from src.dp import run_dp_per_W
from src.dp_opti import run_batch_optimal_dp_per_W
from src.ga import GeneticAlgorithm  
from src.mslb import run_mslb_batch
from src.hungarian_new import run_hungarian_new_per_W
import time
from datetime import datetime
# === 方法選擇 ===
METHOD = "greedy"  # 可選:dp ,greedy ,ga ,mslb,hungarian  ############你要手動設定##############################
# 指定要使用哪組 user 檔 (例如 125 -> data/user_info125.csv)
USER_NUM = 100                     ############你要手動設定##############################
user_csv = f"data/user_info{USER_NUM}.csv"
# === 1️⃣ 載入系統資料 ===
system = load_system_data(regenerate_sat_channels=False, user_csv_path=user_csv)
df_users = system["users"]
df_access = system["access_matrix"]
path_loss = system["path_loss"]
params = system["params"]
sat_channel_dict_backup = system["sat_channel_dict_backup"]
sat_positions = system["sat_positions"]

# 設定 W 與 alpha
W = 2  ############你要手動設定##############################
alpha = params["alpha"]  ############你要手動設定,去src/parameter.py設定##############################

# === 2️⃣ 執行對應的方法 ===
method_start = time.perf_counter()  
if METHOD == "greedy":
    results_df, all_user_paths, load_by_time, df_data_rates = run_greedy_per_W(
        user_df=df_users,
        access_matrix=df_access.to_dict(orient="records"),
        path_loss=path_loss,
        sat_load_dict_backup=copy.deepcopy(sat_channel_dict_backup),
        params=params,
        W=W
    )
elif METHOD == "hungarian":
    results_df, all_user_paths, load_by_time = run_hungarian_per_W(
        df_users=df_users,
        df_access=df_access,
        path_loss=path_loss,
        sat_channel_dict_backup=copy.deepcopy(sat_channel_dict_backup),
        sat_positions=sat_positions,
        params=params,
        W=W
    )
    df_data_rates = results_df.copy()
elif METHOD == "dp":
    results_df, all_user_paths, load_by_time, df_data_rates = run_dp_per_W(
        user_df=df_users,
        access_matrix=df_access.to_dict(orient="records"),
        path_loss=path_loss,
        sat_load_dict_backup=copy.deepcopy(sat_channel_dict_backup),
        params=params,
        W=W
    )
elif METHOD == "ga":
    ga = GeneticAlgorithm(
        population_size=10,
        user_df=df_users,
        access_matrix=df_access.to_dict(orient="records"),
        W=W,
        path_loss=path_loss,
        sat_channel_dict=copy.deepcopy(sat_channel_dict_backup),
        params=params,
        seed=123456  # ✅ 固定一個整數 seed；不想固定就拿掉這行
    )
    ga.evolve(generations=5)  # 訓練 5 輪(5輪大概要1小時)，可調整為 20、50 等等
    results_df, all_user_paths, load_by_time, df_data_rates = ga.export_best_result()
elif METHOD == "mslb":
    results_df, all_user_paths, load_by_time, df_data_rates = run_mslb_batch(
        user_df=df_users,
        access_matrix=df_access.to_dict(orient="records"),
        path_loss=path_loss,
        sat_channel_dict=copy.deepcopy(sat_channel_dict_backup),
        params=params,
        W=W
    )
else:
    raise ValueError(f"未知的 METHOD: {METHOD}")
method_elapsed = time.perf_counter() - method_start 
# === 3️⃣ 重新計算正確的 data rate（考慮所有干擾）===
if isinstance(all_user_paths, pd.DataFrame):
    user_path_records = all_user_paths.to_dict(orient="records")
else:
    user_path_records = all_user_paths

df_correct_rates = recompute_all_data_rates(
    user_path_records, path_loss, params, sat_channel_dict_backup
)

# 與原本 data rate 對齊排序
df_correct_rates = df_correct_rates.set_index(["user_id", "time"]).reindex(
    df_data_rates.set_index(["user_id", "time"]).index
).reset_index()

# === 4️⃣ 建立 results 資料夾 ===
os.makedirs("results", exist_ok=True)
timing_row = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "method": METHOD,
    "W": int(W),
    "alpha": float(alpha),
    "elapsed_sec": round(method_elapsed, 6),
    "num_users": int(len(df_users)),
    "num_time_slots": int(len(df_access)),
}
timing_path = "results/method_timings.csv"
write_header = not os.path.exists(timing_path)
pd.DataFrame([timing_row]).to_csv(
    timing_path, mode="a", header=write_header, index=False
)
# === 5️⃣ 儲存結果 ===
prefix = f"{METHOD}_W{W}_users{USER_NUM}_alpha{alpha}"
results_df.to_csv(f"results/{prefix}_results.csv", index=False)
pd.DataFrame(all_user_paths).to_csv(f"results/{prefix}_paths.csv", index=False)
df_data_rates.to_csv(f"results/{prefix}_data_rates.csv", index=False)

# === 6️⃣ 輸出 Load 狀態（含 initial 隨機 load）===
records = []
df_access = df_access.reset_index(drop=True)
for t in range(len(df_access)):
    visible_sats = df_access.loc[t, "visible_sats"]
    for sat in visible_sats:
        assigned_load = load_by_time[t].get(sat, 0)
        random_load = sum(sat_channel_dict_backup[sat].values()) if sat in sat_channel_dict_backup else 0
        total_load = assigned_load + random_load
        records.append({"time": t, "sat": sat, "load": total_load})
df_load = pd.DataFrame(records)
df_load.to_csv(f"results/{prefix}_load_by_time.csv", index=False)

# === 7️⃣ 輸出正確 data rate（重新計算干擾）===
df_correct_rates.to_csv(f"results/{prefix}_real_data_rates.csv", index=False)

# === 8️⃣ 完成訊息 ===
print(f"\n✅ {METHOD.upper()} 方法完成！")
print(f"📄 結果已儲存至 results/{prefix}_results.csv")
print(f"📄 路徑已儲存至 results/{prefix}_paths.csv")
print(f"📄 Data Rate（當下分配）已儲存至 results/{prefix}_data_rates.csv")
print(f"📄 Load 狀態已儲存至 results/{prefix}_load_by_time.csv")
print(f"📄 Data Rate（重新計算干擾）已儲存至 results/{prefix}_real_data_rates.csv")
