# src/real_reward.py
import os
import glob
import pandas as pd

# ================== 參數設定（自行修改） ==================
W = 2
alpha = 1
folder_path = "results"        # 你的結果都在這裡
out_csv = f"avg_reward_W{W}_alpha{alpha}.csv"
# ========================================================

# 目標：抓像 dp_opti_W2_alpha1_real_data_rates.csv 這類檔名
pattern_data = f"**/*_W{W}_alpha{alpha}_*real_data_rates.csv"
files_data = glob.glob(os.path.join(folder_path, pattern_data), recursive=True)

if not files_data:
    raise FileNotFoundError(
        f"找不到 real_data_rates 檔案：{pattern_data}\n"
        f"請確認 W={W}, alpha={alpha}，以及資料夾：{os.path.abspath(folder_path)}"
    )

def infer_method_name(filepath: str) -> str:
    """從檔名擷取方法名稱（取 _W{W}_ 之前的部分）"""
    base = os.path.basename(filepath)
    key = f"_W{W}_"
    return base.split(key)[0] if key in base else os.path.splitext(base)[0]

def find_load_file_for(data_file: str) -> str | None:
    """
    根據 real_data_rates.csv 的檔名自動找對應的 load_by_time.csv
    1) 先嘗試同資料夾直接替換檔尾
    2) 找不到再用萬用字元掃描（避免中間多了 real_ 等差異）
    """
    ddir = os.path.dirname(data_file)
    base = os.path.basename(data_file)

    # 嘗試 1：同層直接替換尾巴
    cand1 = os.path.join(ddir, base.replace("real_data_rates.csv", "load_by_time.csv"))
    if os.path.exists(cand1):
        return cand1

    # 嘗試 2：萬用字元，確保 method/W/alpha 一致
    method = infer_method_name(data_file)
    wildcard_local = os.path.join(ddir, f"{method}_W{W}_alpha{alpha}_*load_by_time.csv")
    matches = glob.glob(wildcard_local)
    if matches:
        return matches[0]

    # 嘗試 3：整個 results 再找一次（以防不在同層）
    wildcard_global = os.path.join(folder_path, f"**/{method}_W{W}_alpha{alpha}_*load_by_time.csv")
    matches2 = glob.glob(wildcard_global, recursive=True)
    if matches2:
        return matches2[0]

    return None

rows = []

print("找到以下 real_data_rates 檔案：")
for f in files_data:
    print(" -", os.path.relpath(f))

for file_data in files_data:
    method = infer_method_name(file_data)
    load_file = find_load_file_for(file_data)

    if not load_file:
        print(f"[警告] {method} 找不到對應的 load_by_time 檔，跳過")
        continue

    # 讀資料
    df_data = pd.read_csv(file_data)
    df_load = pd.read_csv(load_file)

    # 欄位檢查
    need_data_cols = {"user_id", "time", "sat", "channel", "data_rate"}
    need_load_cols = {"time", "sat", "load"}
    if not need_data_cols.issubset(df_data.columns):
        print(f"[警告] {method} 的 data_rates 缺欄位：{need_data_cols}，實際={list(df_data.columns)}，跳過")
        continue
    if not need_load_cols.issubset(df_load.columns):
        print(f"[警告] {method} 的 load_by_time 缺欄位：{need_load_cols}，實際={list(df_load.columns)}，跳過")
        continue

    # 合併 (time, sat)
    merged = pd.merge(df_data, df_load, on=["time", "sat"], how="left")

    # 若有缺 load（理論上不該有），先補 0 避免 NaN
    if merged["load"].isna().any():
        missing = int(merged["load"].isna().sum())
        print(f"[提示] {method} 有 {missing} 筆找不到 load，已當作 0 處理")
        merged["load"] = merged["load"].fillna(0.0)

    # Reward = (1 - alpha * L) * data_rate，注意:我們假設每個衛星都有25個channel,若你有設定channel數 != 25，請自行調整
    merged["reward"] = (1 - alpha * merged["load"] / 25) * merged["data_rate"]

    # === 儲存合併後的資料到 results/merged/ ===
    os.makedirs("results/merged", exist_ok=True)
    merged_out_path = f"results/merged/{method}_W{W}_alpha{alpha}_merged.csv"
    merged.to_csv(merged_out_path, index=False)
    print(f"[儲存] 合併後資料寫入：{merged_out_path}")


    # 兩種平均：整體平均 & 先 per-user 再平均（更公平）
    overall_avg_reward = merged["reward"].mean()
    per_user_avg_reward = merged.groupby("user_id")["reward"].mean().mean()

    # 也順手算平均 data_rate 供對照
    overall_avg_rate = merged["data_rate"].mean()
    per_user_avg_rate = merged.groupby("user_id")["data_rate"].mean().mean()

    rows.append({
        "method": method,
        "avg_reward_overall": overall_avg_reward,
        "avg_reward_per_user": per_user_avg_reward,
        "avg_rate_overall": overall_avg_rate,
        "avg_rate_per_user": per_user_avg_rate,
        "data_file": os.path.relpath(file_data),
        "load_file": os.path.relpath(load_file)
    })

# 輸出結果
if not rows:
    print("沒有可用的方法可計算，請確認 *_real_data_rates.csv 與 *_load_by_time.csv 是否對應齊全。")
else:
    df_out = pd.DataFrame(rows)

    # 依常見方法排序（未列到的放最後）
    preferred = ["dp_opti", "dp", "ga", "greedy", "hungarian", "mslb"]
    df_out["order"] = df_out["method"].apply(lambda m: preferred.index(m) if m in preferred else 999)
    df_out = df_out.sort_values(["order", "method"]).drop(columns=["order"])

    print("\n=== 平均 Reward 結果 ===")
    print(df_out[["method", "avg_reward_overall", "avg_reward_per_user"]])

    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n已輸出：{out_csv}")
