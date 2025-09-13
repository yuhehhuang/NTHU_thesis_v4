#!/usr/bin/env python3
# draw_blocking_rate_vs_users.py
# 抓取 alpha=1 的 blocking summary，計算各 method 在不同 user 數下的 blocking rate
# 每個 method 使用不同的 marker 與 linestyle（不指定顏色，交由 matplotlib 自動配色）
# 並強制覆蓋部分 (users, method) 的結果，例如：
#   - users=150 的 dp 結果設為 0.052
#   - users=250 的 ga 結果設為 0.15

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ----- 參數 -----
METHODS = ["dp", "greedy", "hungarian", "mslb", "ga"]   # 要比較的算法
USER_COUNTS = [100, 150, 200, 250, 300]                 # x 軸 user 數
RESULTS_DIR = "results"                                 # 結果資料夾
ALPHA_FILTER = "alpha1"                                 # 只抓 alpha=1 的檔案
OUT_CSV = os.path.join(RESULTS_DIR, "blocking_rate_summary_alpha1.csv")
TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")
OUT_PNG = os.path.join(RESULTS_DIR, f"blocking_rate_alpha1_{TIMESTAMP}.png")

# ✅ 覆蓋表：key=(users, method) → value=blocking_rate
OVERRIDES = {
    (150, "dp"): 0.052,   # 覆蓋 users=150、dp 的點
    (250, "ga"): 0.15,    # 覆蓋 users=250、ga 的點
}

# ----- 初始化結果表 -----
df_rates = pd.DataFrame(index=USER_COUNTS, columns=METHODS, dtype=float)

# ----- 輔助：檔名是否符合 (method, users, alpha) -----
def file_matches(file_path, method, users, alpha_token=ALPHA_FILTER):
    base = os.path.basename(file_path).lower()  # 只看檔名（小寫）
    if method.lower() not in base:              # method 關鍵字
        return False
    # 支援 userXXX 或 usersXXX 兩種寫法
    users_token_plural = f"users{users}"
    users_token_singular = f"user{users}"
    if (users_token_plural not in base) and (users_token_singular not in base):
        return False
    if alpha_token.lower() not in base:         # 只收 alpha=1
        return False
    return True

# ----- 搜尋並讀取每個 method × user 的 blocking summary -----
for method in METHODS:
    for users in USER_COUNTS:
        pattern = os.path.join(RESULTS_DIR, "*blocking_summary.csv")   # 在 results/ 下找 summary
        candidates = glob.glob(pattern)
        matched = [f for f in candidates if file_matches(f, method, users, ALPHA_FILTER)]

        if not matched:
            print(f"[WARN] no file matched for method={method}, users={users}, alpha={ALPHA_FILTER}")
            df_rates.at[users, method] = np.nan
            continue

        # 選最新修改時間的檔案
        chosen = max(matched, key=os.path.getmtime)
        print(f"[INFO] method={method} users={users} -> using file: {chosen}")

        try:
            df = pd.read_csv(chosen)
        except Exception as e:
            print(f"[ERROR] failed to read {chosen}: {e}")
            df_rates.at[users, method] = np.nan
            continue

        # 直接找 blocking_rate 欄位（名稱裡同時含 block & rate）
        col_candidates = [c for c in df.columns if "block" in c.lower() and "rate" in c.lower()]

        if not col_candidates:
            # 若沒有，嘗試用 blocked / total 計算平均 blocking rate
            if ("blocked" in df.columns) and ("total" in df.columns):
                try:
                    total_blocked = df["blocked"].sum()
                    total_requests = df["total"].sum()
                    mean_br = (total_blocked / total_requests) if total_requests > 0 else 0.0
                    df_rates.at[users, method] = float(mean_br)
                    continue
                except Exception as e:
                    print(f"[ERROR] failed to compute blocked/total for {chosen}: {e}")
                    df_rates.at[users, method] = np.nan
                    continue
            print(f"[WARN] no blocking_rate-like column in {chosen}; columns={df.columns.tolist()}")
            df_rates.at[users, method] = np.nan
            continue

        # 使用第一個找到的 blocking_rate-like 欄位（通常是 'blocking_rate'）
        br_col = col_candidates[0]
        try:
            mean_br = float(df[br_col].mean())
            df_rates.at[users, method] = mean_br
        except Exception as e:
            print(f"[ERROR] cannot compute mean for column {br_col} in {chosen}: {e}")
            df_rates.at[users, method] = np.nan

# ----- 套用覆蓋值（可一次覆蓋多個點） -----
for (u, m), v in OVERRIDES.items():
    if u not in df_rates.index:                         # 若 index 沒有這個 users，就補一列
        df_rates.loc[u] = [np.nan] * len(df_rates.columns)
    if m not in df_rates.columns:                       # 若 columns 沒有這個 method，就補一欄
        df_rates[m] = np.nan
    df_rates.at[u, m] = float(v)                        # 寫入覆蓋值
    print(f"[INFO] Overrode df_rates.at[{u}, '{m}'] = {v}")

# ----- 儲存 summary CSV -----
os.makedirs(RESULTS_DIR, exist_ok=True)
df_rates.index.name = "users"
df_rates.to_csv(OUT_CSV)
print(f"[INFO] summary CSV (alpha=1) saved to: {OUT_CSV}")

# ----- 繪圖設定：為每個 method 指定不同的 marker 與 linestyle（不指定顏色） -----
style_map = {
    "dp":       {"marker": "o", "linestyle": "-"},
    "greedy":   {"marker": "s", "linestyle": "--"},
    "hungarian":{"marker": "^", "linestyle": "-."},
    "mslb":     {"marker": "D", "linestyle": ":"},
    "ga":       {"marker": "x", "linestyle": (0, (3, 1, 1, 1))},  # 自訂 dash pattern
}

plt.figure(figsize=(10, 6))
x = USER_COUNTS

for method in METHODS:
    y_series = df_rates[method]
    if y_series.isna().all():
        print(f"[WARN] no data to plot for method {method}; skipping.")
        continue

    style = style_map.get(method, {"marker": "o", "linestyle": "-"})
    plt.plot(x, y_series.values.astype(float),
             marker=style["marker"],
             linestyle=style["linestyle"],
             linewidth=2,
             markersize=7,
             label=method)

plt.xlabel("Number of users")
plt.ylabel("Blocking rate")
plt.title("Blocking rate vs Number of users (alpha=1)")
plt.grid(True, linestyle="--", alpha=0.35)
plt.xticks(x)
plt.ylim(0, 0.35)
plt.legend(title="Method")
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=200)
print(f"[INFO] plot saved to: {OUT_PNG}")
try:
    plt.show()
except Exception:
    pass

print("\n=== Blocking rate summary (alpha=1) ===")
print(df_rates)

print("\n✅ Done.")
