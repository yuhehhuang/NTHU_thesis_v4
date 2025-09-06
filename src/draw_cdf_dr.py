import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==== 參數設定 ====
W = 2
alpha = 1
folder_path = "results"
alpha_symbol = "\u03B1"
pattern = f"**/*_W{W}_alpha{alpha}_*data_rates.csv"
files = glob.glob(os.path.join(folder_path, pattern), recursive=True)

# ✅ 只保留這些方法
selected_methods = ["dp", "ga", "greedy", "mslb", "hungarian"]

def infer_method_name(filepath: str) -> str:
    base = os.path.basename(filepath)
    if f"_W{W}_" in base:
        return base.split(f"_W{W}_")[0]
    return os.path.splitext(base)[0]

# ==== 蒐集每個方法的「每個 user 的平均 data rate」 ====
method_user_rates = {}

for file in files:
    method = infer_method_name(file)
    if method not in selected_methods:
        continue  # ❌ 跳過未選擇的方法

    df = pd.read_csv(file)
    if not {"user_id", "data_rate"}.issubset(df.columns):
        continue


    # 統計每個 user 的平均
    user_avg = df.groupby("user_id")["data_rate"].mean().sort_values().reset_index(drop=True)
    method_user_rates[method] = user_avg

# ==== 繪製 CDF 圖 ====
plt.figure(figsize=(10, 6))
for method in selected_methods:
    if method not in method_user_rates:
        continue
    data = method_user_rates[method]
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=method, marker='.', linestyle='-')

plt.title(f"CDF of Per-User Average Data Rate (W={W}, {alpha_symbol}={alpha})", fontsize=14)
plt.xlabel("Per-User Average Data Rate", fontsize=12)
plt.ylabel("Cumulative Proportion", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"cdf_user_data_rate_W{W}_{alpha_symbol}{alpha}_filtered.png", dpi=300)
plt.show()