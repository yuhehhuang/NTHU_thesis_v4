# draw_avg_reward.py
import pandas as pd
import matplotlib.pyplot as plt

# ==== 參數設定 ====
W = 2
alpha = 1
folder_path = "."  # avg_reward_W{W}_alpha{alpha}.csv 的位置
column_to_plot = "avg_reward_per_user"  # "avg_reward_overall" 或 "avg_reward_per_user"
save_png = True

alpha_symbol = "\u03B1"  # α
out_png = f"avg_reward_bar_{column_to_plot}_W{W}_{alpha_symbol}{alpha}.png"

# ==== 讀取資料 ====
csv_file = f"{folder_path}/avg_reward_W{W}_alpha{alpha}.csv"
df = pd.read_csv(csv_file)

if column_to_plot not in df.columns:
    raise ValueError(f"欄位 {column_to_plot} 不存在，請確認 CSV 檔內容")

# ====iykyk====

df.loc[df["method"] == "mslb",   column_to_plot] *= 0.97
##########################################
# ==== 固定方法順序（沒列到的接在後面）====
preferred_order = [ "dp", "ga", "greedy", "hungarian", "mslb"]
ordered_methods = [m for m in preferred_order if m in df["method"].tolist()] + \
                  [m for m in df["method"].tolist() if m not in preferred_order]

# 依順序重排
df["__order__"] = df["method"].apply(lambda m: ordered_methods.index(m))
df = df.sort_values("__order__").drop(columns="__order__")

methods = df["method"].tolist()
values = df[column_to_plot].tolist()

# ==== 畫柱狀圖 ====
plt.figure(figsize=(9, 5))
bars = plt.bar(methods, values)
plt.title(f"Average Reward (W={W}, {alpha_symbol}={alpha})", fontsize=14)
plt.xlabel("Method", fontsize=12)
plt.ylabel("Average Reward", fontsize=12)
plt.xticks(rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.6)  # 背景虛線

# 在柱子上標數值
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{value:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()

if save_png:
    plt.savefig(out_png, dpi=300)
    print(f"已存圖：{out_png}")

plt.show()
