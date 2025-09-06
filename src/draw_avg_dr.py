import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==== 參數設定（可自行修改）====
W = 2
alpha = 1
folder_path = "results"  # 結果資料夾
save_png = True
save_csv = True
alpha_symbol = "\u03B1"  # α 的 Unicode
out_png = f"avg_user_data_rate_W{W}_{alpha_symbol}{alpha}.png"
out_csv = f"avg_user_data_rate_W{W}_{alpha_symbol}{alpha}.csv"

# ==== 檔案搜尋 ====
pattern = f"**/*_W{W}_alpha{alpha}_*data_rates.csv"
files = glob.glob(os.path.join(folder_path, pattern), recursive=True)

if not files:
    raise FileNotFoundError(
        f"找不到符合樣式的檔案：{pattern}\n"
        f"請確認 W={W}, alpha={alpha}，以及檔案是否在 '{os.path.abspath(folder_path)}' 裡。"
    )

print("找到以下檔案：")
for f in files:
    print(" -", os.path.relpath(f))

# ==== 幫助函式 ====
#把整條路徑只取出「檔名本身」
#filepath = "results/dp_opti_W2_alpha1_real_data_rates.csv"
#base = "dp_opti_W2_alpha1_real_data_rates.csv"
####################################################
def infer_method_name(filepath: str) -> str:
    base = os.path.basename(filepath)
    if f"_W{W}_" in base:
        return base.split(f"_W{W}_")[0]
    return os.path.splitext(base)[0]

# ==== 計算每個方法的「平均 user data rate」====
preferred_order = ["dp", "ga", "greedy", "hungarian", "mslb"]
method_to_avg = {}

for file in files:
    method = infer_method_name(file)
    df = pd.read_csv(file)
    required_cols = {"user_id", "time", "sat", "channel", "data_rate"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{file} 缺少必要欄位 {required_cols}，實際欄位={list(df.columns)}"
        )

    per_user_mean = df.groupby("user_id")["data_rate"].mean()
    method_avg = per_user_mean.mean()
    method_to_avg[method] = float(method_avg)

# ==== 上帝之手調整 ====

# ==== 排序 ====
ordered_methods = []
for m in preferred_order:
    if m in method_to_avg:
        ordered_methods.append(m)
for m in method_to_avg:
    if m not in ordered_methods:
        ordered_methods.append(m)

avg_values = [method_to_avg[m] for m in ordered_methods]

# ==== 輸出 CSV ====
if save_csv:
    df_out = pd.DataFrame({
        "method": ordered_methods,
        "avg_user_data_rate": avg_values
    })
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已輸出彙整表：{out_csv}")

# ==== 畫柱狀圖 ====
plt.figure(figsize=(9, 5))
bars = plt.bar(ordered_methods, avg_values)
plt.title(f"Average User Data Rate per Method (W={W}, {alpha_symbol}={alpha})", fontsize=14)
plt.xlabel("Method", fontsize=12)
plt.ylabel("Average User Data Rate (Mbps)", fontsize=12)
plt.xticks(rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.6)  # 背景虛線

# 在柱子上顯示數值
for bar, value in zip(bars, avg_values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{value:.2f}",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()

if save_png:
    plt.savefig(out_png, dpi=300)
    print(f"已存圖：{out_png}")

plt.show()
