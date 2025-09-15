#!/usr/bin/env python3
# draw_blocking_rate_vs_alpha.py
# 針對 W=3, users=100，繪製不同 alpha 值下各 method 的 blocking rate（x=alpha, y=blocking rate）

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- 使用設定 ----------
RESULTS_DIR = "results"                         # results 資料夾
W_VALUE = 2                                     # 固定 W = 3
USERS = 100                                     # 固定 user = 100
METHODS = ["dp", "greedy", "hungarian", "mslb", "ga"]  # 要畫的 method 列表
OUT_CSV = os.path.join(RESULTS_DIR, f"blocking_rate_W{W_VALUE}_users{USERS}_by_alpha.csv")
TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")
OUT_PNG = os.path.join(RESULTS_DIR, f"blocking_rate_W{W_VALUE}_users{USERS}_by_alpha_{TIMESTAMP}.png")
# ---------- 幫助函式 ----------
def extract_alpha_from_filename(fname):
    """
    從檔名擷取 alpha 的數值（返回 float），若找不到則回傳 None。
    支援的格式例子： 'alpha1', 'alpha_1', 'alpha-1', 'alpha0.5', 'alpha_0.5'
    """
    b = os.path.basename(fname).lower()
    # regex: alpha[_-]?digits(.digits)?
    m = re.search(r"alpha[_\-]?([0-9]+(?:\.[0-9]+)?)", b)
    if not m:
        # 也嘗試 α 或 Α 的情況（有些檔名可能包含 unicode alpha）
        m2 = re.search(r"[α\u0391][_ -]?([0-9]+(?:\.[0-9]+)?)", b)
        if m2:
            try:
                return float(m2.group(1))
            except Exception:
                return None
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def detect_method_from_filename(fname, methods=METHODS):
    """
    根據 method 名字在檔名中出現的情況來判斷 method（返回 method 字串或 None）。
    以 methods 的順序為優先（避免部分名稱互爭）。
    """
    b = os.path.basename(fname).lower()
    for m in methods:
        if m.lower() in b:
            return m
    return None

def compute_mean_blocking_rate_from_file(path):
    """
    嘗試讀取 blocking summary csv 並計算 mean blocking rate。
    優先尋找 column name 包含 'block' 與 'rate' 的欄位，
    否則嘗試用 blocked/total 計算 (sum(blocked)/sum(total))。
    回傳 float（或 None 若無法計算）。
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] failed to read {path}: {e}")
        return None

    # 找包含 block & rate 的欄位
    cols = [c for c in df.columns if ("block" in c.lower()) and ("rate" in c.lower())]
    if cols:
        try:
            # 若欄位是逐行的 blocking rate，取平均；若已是單一值也沒問題
            return float(pd.to_numeric(df[cols[0]], errors="coerce").mean())
        except Exception:
            return None

    # 否則嘗試 blocked/total
    if ("blocked" in df.columns) and ("total" in df.columns):
        try:
            tot_blocked = pd.to_numeric(df["blocked"], errors="coerce").sum()
            tot = pd.to_numeric(df["total"], errors="coerce").sum()
            return float(tot_blocked / tot) if tot > 0 else 0.0
        except Exception:
            return None

    return None

# ---------- 收集 candidate 檔案 ----------
# 我們限定 filename 同時包含 W{W_VALUE} 與 users{USERS}（支援 user/users）
pattern1 = os.path.join(RESULTS_DIR, f"*W{W_VALUE}*users{USERS}*blocking_summary.csv")
pattern2 = os.path.join(RESULTS_DIR, f"*W{W_VALUE}*user{USERS}*blocking_summary.csv")
candidates = glob.glob(pattern1) + glob.glob(pattern2)

if not candidates:
    print(f"[WARN] no blocking_summary files found for W{W_VALUE} users{USERS} in {RESULTS_DIR}")
    # still create empty csv & empty plot placeholder
    pd.DataFrame().to_csv(OUT_CSV)
    print("Exiting.")
    raise SystemExit(0)

# ---------- 解析檔案並聚合 data ----------
# data structure: dict[method][alpha] = (chosen_file_path, mean_blocking_rate)
results = {m: {} for m in METHODS}
alpha_set = set()

# 如果同 (method,alpha) 有多個檔案，選最新（mtime）
for p in candidates:
    alpha = extract_alpha_from_filename(p)
    if alpha is None:
        # skip files we cannot infer alpha from
        continue
    method = detect_method_from_filename(p)
    if method is None:
        # skip files that do not match desired methods
        continue
    # only consider methods in our METHODS list
    if method not in METHODS:
        continue
    alpha_set.add(alpha)
    # compute blocking rate
    br = compute_mean_blocking_rate_from_file(p)
    if br is None:
        # skip if cannot calculate
        continue
    # if we already have an entry for this (method,alpha) compare mtimes and keep newest
    prev = results[method].get(alpha)
    mtime = os.path.getmtime(p)
    if prev is None:
        results[method][alpha] = (p, br, mtime)
    else:
        # prev is (path, br, mtime)
        if mtime > prev[2]:
            results[method][alpha] = (p, br, mtime)

# ---------- 準備 DataFrame（index=alpha sorted, columns=methods） ----------
if not alpha_set:
    print("[WARN] no alpha values parsed from candidate filenames. Exiting.")
    raise SystemExit(0)

alphas_sorted = sorted(list(alpha_set))
df_rates = pd.DataFrame(index=alphas_sorted, columns=METHODS, dtype=float)

# fill df_rates from results
for method in METHODS:
    for a in alphas_sorted:
        entry = results.get(method, {}).get(a)
        if entry is None:
            df_rates.at[a, method] = np.nan
        else:
            df_rates.at[a, method] = float(entry[1])

# ---------- 寫出 CSV（alpha 為 index） ----------
os.makedirs(RESULTS_DIR, exist_ok=True)
df_rates.index.name = "alpha"
df_rates.to_csv(OUT_CSV)
print(f"[INFO] saved summary CSV to: {OUT_CSV}")

# ---------- 繪圖：x = alpha (數值)，y = blocking rate ----------
plt.figure(figsize=(9,6))
x = alphas_sorted

# 指定繪圖順序與樣式（包含顏色）
desired_order = ["dp", "ga", "greedy", "hungarian", "mslb"]
style_map = {
    "dp":        {"marker":"o", "linestyle":"-",  "color":"tab:blue"},
    "ga":        {"marker":"s", "linestyle":"--", "color":"tab:orange"},
    "greedy":    {"marker":"^", "linestyle":"-.", "color":"tab:green"},
    "hungarian": {"marker":"D", "linestyle":":",  "color":"tab:red"},
    "mslb":      {"marker":"v", "linestyle":(0, (3,1,1,1)), "color":"tab:purple"}
}

# 依 desired_order 畫圖（若 METHODS 中沒有也會自動跳過）
# 先建立 col map（處理大小寫可能不同的情況）
colname_map = {c.lower(): c for c in df_rates.columns}

for m in desired_order:
    if m not in colname_map:
        continue
    col_actual = colname_map[m]
    # safe extraction：轉 float 並保留 NaN
    y = pd.to_numeric(df_rates[col_actual], errors="coerce").values.astype(float)
    if np.all(np.isnan(y)):
        print(f"[WARN] no data for method {m}, skipping plot.")
        continue
    style = style_map.get(m, {"marker":"o","linestyle":"-","color":None})
    plt.plot(x, y, label=m, marker=style["marker"], linestyle=style["linestyle"],
             color=style.get("color"), linewidth=2, markersize=6)

plt.xlabel("alpha")
plt.ylabel("Blocking rate")
plt.title(f"Blocking rate vs alpha (W={W_VALUE}, users={USERS})")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(title="Method", loc="best")
plt.tight_layout()
plt.ylim(0, 0.15)

# save & show
plt.savefig(OUT_PNG, dpi=200)
print(f"[INFO] saved plot to: {OUT_PNG}")
try:
    plt.show()
except Exception:
    pass

# ---------- 印出表格以便檢查 ----------
print("\n=== Blocking rate by alpha ===")
print(df_rates)

print("\n✅ Done.")
