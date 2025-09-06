import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==== 參數設定 ====
W = 2
alpha = 1
folder_path = "results"  # 結果資料夾
save_png = True
alpha_symbol = "\u03B1"  # α
out_png = f"avg_load_variance_W{W}_{alpha_symbol}{alpha}.png"

# ==== 檔案搜尋 ====
pattern = f"**/*_W{W}_alpha{alpha}_*load_by_time.csv"
files = glob.glob(os.path.join(folder_path, pattern), recursive=True)

if not files:
    raise FileNotFoundError(f"找不到符合 W={W}, alpha={alpha} 的檔案")

print("找到以下檔案：")
for f in files:
    print(" -", os.path.relpath(f))

# ==== 計算每個方法的平均 variance ====
method_variances = {}
for file in files:
    base = os.path.basename(file)
    method_name = base.split(f"_W{W}")[0]  # 取 _W 之前的字串

    df = pd.read_csv(file)
    required_cols = {"time", "sat", "load"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{file} 缺少必要欄位 {required_cols}")

    var_per_t = df.groupby("time")["load"].var(ddof=0).fillna(0.0)
    avg_var = var_per_t.mean()
    method_variances[method_name] = avg_var

# ==== 固定方法順序 ====
preferred_order = [ "dp", "ga", "greedy", "hungarian", "mslb"]
methods = [m for m in preferred_order if m in method_variances] + \
          [m for m in method_variances if m not in preferred_order]
avg_vars = [method_variances[m] for m in methods]

# ==== 畫折線圖 ====
plt.figure(figsize=(9, 5))
plt.plot(methods, avg_vars, marker='o', linestyle='-')
plt.title(f"Average Load Variance (W={W}, {alpha_symbol}={alpha})", fontsize=14)
plt.xlabel("Method", fontsize=12)
plt.ylabel("Average Load Variance", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=20)

# 在點上顯示數值
for x, y in zip(methods, avg_vars):
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()

if save_png:
    plt.savefig(out_png, dpi=300)
    print(f"已存圖：{out_png}")

plt.show()
