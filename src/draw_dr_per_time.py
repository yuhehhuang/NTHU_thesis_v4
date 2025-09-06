import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==== 參數設定 ====
W = 2
alpha = 1
folder_path = "results"
save_png = True
alpha_symbol = "\u03B1"
out_png = f"total_system_throughput_W{W}_{alpha_symbol}{alpha}.png"

# ==== 只顯示這幾種方法 ====
methods_to_plot = ["dp", "ga", "mslb", "greedy", "hungarian"]

# ==== 搜尋符合條件的檔案 ====
pattern = f"**/*_W{W}_alpha{alpha}_*data_rates.csv"
files = glob.glob(os.path.join(folder_path, pattern), recursive=True)

if not files:
    raise FileNotFoundError(f"❌ 找不到符合樣式的檔案：{pattern}")

print("✅ 找到以下檔案：")
for f in files:
    print(" -", os.path.relpath(f))

# ==== 幫助函式 ====
def infer_method_name(filepath: str) -> str:
    base = os.path.basename(filepath)
    if f"_W{W}_" in base:
        return base.split(f"_W{W}_")[0]
    return os.path.splitext(base)[0]

# ==== 儲存每個方法的 time-series throughput ====
method_to_series = {}

for file in files:
    method = infer_method_name(file)

    if method not in methods_to_plot:
        continue  # 忽略不在指定範圍內的方法

    df = pd.read_csv(file)
    required_cols = {"user_id", "time", "sat", "channel", "data_rate"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{file} 缺少必要欄位 {required_cols}，實際欄位={list(df.columns)}")

    # 計算每個 time slot 的總 throughput
    series = df.groupby("time")["data_rate"].sum().sort_index()


    method_to_series[method] = series

# ==== 排序與 marker 樣式 ====
methods = [m for m in methods_to_plot if m in method_to_series]

method_markers = {
    "dp_opti": "o",
    "ga": "D",
    "mslb": "X",
    "greedy": "^",
    "hungarian_new": "P",
}

# ==== 畫圖 ====
plt.figure(figsize=(10, 6))

for method in methods:
    series = method_to_series[method]
    marker = method_markers.get(method, 'o')
    plt.plot(series.index, series.values, label=method, marker=marker, linewidth=1.5)

plt.title(f"Total System Throughput per Time Slot\n(W={W}, {alpha_symbol}={alpha})", fontsize=14)
plt.xlabel("Time Slot", fontsize=12)
plt.ylabel("Total Data Rate (Mbps)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

if save_png:
    plt.savefig(out_png, dpi=300)
    print(f"✅ 已儲存圖片：{out_png}")

plt.show()
