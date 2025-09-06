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
out_png = f"variance_over_time_W{W}_{alpha_symbol}{alpha}.png"

# 只保留這些方法
methods_to_plot = ["dp", "greedy", "hungarian", "mslb", "ga"]

# ==== 搜尋檔案 ====
pattern = f"**/*_W{W}_alpha{alpha}_*load_by_time.csv"
files = glob.glob(os.path.join(folder_path, pattern), recursive=True)

if not files:
    raise FileNotFoundError(f"找不到符合 W={W}, alpha={alpha} 的檔案")

print("找到以下檔案：")
for f in files:
    print(" -", os.path.relpath(f))

# ==== 計算每個方法的變異數 series ====
method_to_variance_series = {}

for file in files:
    base = os.path.basename(file)
    method_name = base.split(f"_W{W}")[0]

    if method_name not in methods_to_plot:
        continue

    df = pd.read_csv(file)
    if not {"time", "sat", "load"}.issubset(df.columns):
        raise ValueError(f"{file} 缺少必要欄位")

    # 每個時間點的負載變異數
    var_series = df.groupby("time")["load"].var(ddof=0).fillna(0.0)
    method_to_variance_series[method_name] = var_series

# ==== 畫圖 ====
method_markers = {
    "dp_opti": "o",
    "greedy": "^",
    "hungarian_new": "P",
    "mslb": "X",
    "ga": "D",
}

plt.figure(figsize=(10, 6))

for method in methods_to_plot:
    if method in method_to_variance_series:
        series = method_to_variance_series[method]
        marker = method_markers.get(method, 'o')
        plt.plot(series.index, series.values, label=method, marker=marker, linewidth=1.5)

plt.title(f"Load Variance per Time Slot (W={W}, {alpha_symbol}={alpha})", fontsize=14)
plt.xlabel("Time Slot", fontsize=12)
plt.ylabel("Load Variance", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

if save_png:
    plt.savefig(out_png, dpi=300)
    print(f"✅ 已儲存圖片：{out_png}")

plt.show()
