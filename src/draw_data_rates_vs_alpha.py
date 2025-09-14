#!/usr/bin/env python3
# src/draw_data_rates_vs_alpha.py
# 從「專案根」遞迴搜尋所有資料夾中的 avg_user_data_rate_* 檔案，
# 解析 alpha 與 method，繪製 x=alpha, y=avg user data rate（dp=實線，其餘=虛線），
# 並把圖與彙總 CSV 存到 <專案根>/results/。

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======== 可調參數 ========
DEFAULT_W = 2
DEFAULT_USERS = 100
OUT_DPI = 150

# ==========================

# 路徑設定：從專案根遞迴掃描；輸出固定到 <專案根>/results/
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))          # src/
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))    # 專案根
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def find_rate_col(cols):
    """猜測 data rate 欄位；若沒找到就回退到『所有數值欄平均』"""
    cols_low = [c.lower() for c in cols]
    for i, c in enumerate(cols_low):
        if re.search(r'avg.*data.*rate', c) or re.search(r'avg.*user.*rate', c) or re.search(r'avg.*rate', c):
            return cols[i]
    numeric_candidates = []
    for c in cols:
        if c.lower() in ("time", "t", "alpha", "user_id", "userid", "users", "method"):
            continue
        numeric_candidates.append(c)
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    return None

def parse_alpha_from_name(name):
    """從檔名解析 alpha（支援 α/alpha/a）"""
    normalized = name.replace('Α', 'alpha').replace('α', 'alpha')
    m = re.search(r'alpha[_\-]?([0-9]+(?:\.[0-9]+)?)', normalized, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m2 = re.search(r'[_\-]a([0-9]+(?:\.[0-9]+)?)', name, re.IGNORECASE)
    if m2:
        return float(m2.group(1))
    return None

def recursive_find_files(root_folder, patterns):
    """從 root_folder 遞迴搜尋符合 patterns 的檔案，並去重"""
    found = []
    for pat in patterns:
        fullpat = os.path.join(root_folder, "**", pat)
        matches = glob.glob(fullpat, recursive=True)
        found.extend(matches)
    # 去重並排序
    found = sorted(list(dict.fromkeys(found)))
    return found

def build_style_list():
    """產生 (marker, linestyle) 清單"""
    markers = ['o','s','^','D','v','<','>','p','h','x','+','*','1','2','3','4']
    linestyles = ['-','--','-.',':']
    return [(mk, ls) for ls in linestyles for mk in markers]

def is_alpha_like_series(series, alpha_value):
    """避免把常數 alpha 欄誤當作 rate"""
    try:
        vals = pd.to_numeric(series, errors='coerce').dropna()
    except Exception:
        return False
    if vals.empty:
        return False
    if np.allclose(vals.values, np.full(len(vals), alpha_value), atol=1e-6):
        return True
    if vals.std() < 1e-6 and abs(vals.mean() - alpha_value) < 1e-6:
        return True
    if vals.nunique() == 1 and abs(vals.iloc[0] - alpha_value) < 1e-6:
        return True
    return False

def main():
    W = DEFAULT_W
    USERS = DEFAULT_USERS

    print("Search root:", PROJECT_ROOT)

    # 從專案根遞迴找所有檔案（不管放哪個資料夾）
    patterns = [
        f"avg_user_data_rate_W{W}_users{USERS}_*.csv",
        f"avg_user_data_rate_*W{W}*_users{USERS}_*alpha*.csv",
        f"avg_user_data_rate_*W{W}*_users{USERS}_*α*.csv",
    ]
    found = recursive_find_files(PROJECT_ROOT, patterns)

    # 避免把自己產生的 summary 檔再讀進來
    found = [p for p in found if "summary" not in os.path.basename(p).lower()]

    print(f"Total files found: {len(found)}")
    for f in found[:50]:
        print(" -", os.path.relpath(f, PROJECT_ROOT))
    if not found:
        print("沒有找到檔案。請確認檔名包含 'avg_user_data_rate'。")
        return

    data = {}          # data[method][alpha] = value
    alphas_set = set()

    for fpath in found:
        fname = os.path.basename(fpath)

        # 1) 從檔名抓 alpha；不行再嘗試讀 csv 的 alpha 欄
        alpha = parse_alpha_from_name(fname)
        if alpha is None:
            try:
                tmp = pd.read_csv(fpath, nrows=5)
                for c in tmp.columns:
                    if c.lower() == 'alpha':
                        col = pd.read_csv(fpath, usecols=[c]).dropna()
                        if not col.empty:
                            alpha = float(col.iloc[0, 0])
                            break
            except Exception:
                pass
        if alpha is None:
            # 沒 alpha 的檔案對此圖無用（因為 x 軸是 alpha）
            continue

        # 2) 讀檔
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print("讀檔失敗:", fname, e)
            continue
        if df.empty:
            continue

        # 3) 找 method 與 rate；否則回退到數值欄平均
        method_col = None
        for cand in ("method", "Method", "METHOD"):
            if cand in df.columns:
                method_col = cand
                break
        rate_col = find_rate_col(list(df.columns))

        if method_col and rate_col:
            grp = df.groupby(method_col)[rate_col].mean()
            for method, val in grp.items():
                if pd.notna(val):
                    data.setdefault(str(method), {})[alpha] = float(val)
                    alphas_set.add(alpha)
        else:
            # 回退：對「合理的數值欄」取平均，視為 single method
            skip_names = {"alpha","a","w","users","user","time","t","id","index","method"}
            numeric_cols = []
            for c in df.columns:
                if c.lower() in skip_names:
                    continue
                if re.search(r'\balpha\b', c, re.IGNORECASE) or 'α' in c:
                    continue
                ser = pd.to_numeric(df[c], errors='coerce').dropna()
                if ser.empty:
                    continue
                if is_alpha_like_series(ser, alpha):
                    continue
                numeric_cols.append(c)

            if numeric_cols:
                # ✅ 修正：逐欄轉數字，單欄直接平均，多欄攤成 numpy 取 nanmean
                if len(numeric_cols) == 1:
                    ser = pd.to_numeric(df[numeric_cols[0]], errors='coerce')
                    val = float(ser.mean(skipna=True))
                else:
                    vals_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                    val = float(np.nanmean(vals_df.to_numpy()))
                if not np.isnan(val):
                    data.setdefault("unknown_method", {})[alpha] = val
                    alphas_set.add(alpha)

    if not data:
        print("沒有可用的數據被擷取。")
        return

    # 4) 彙整：alpha x method 表
    methods = sorted(data.keys())
    alphas = sorted(alphas_set)
    result = pd.DataFrame(index=alphas, columns=methods, dtype=float)
    for m in methods:
        for a, v in data[m].items():
            result.at[a, m] = v
    result.index.name = "alpha"
    print("\n彙總表 (alpha x method):")
    print(result)

    # 5) 作圖（dp=實線；其他=虛線；每 method 不同 marker）
    styles = build_style_list()
    plt.figure(figsize=(10,6))
    xs = result.index.values
    for i, col in enumerate(result.columns):
        ys = result[col].values.astype(float)
        if np.all(np.isnan(ys)):
            continue
        mk = styles[i % len(styles)][0]
        ls = '-' if 'dp' in col.lower() else '--'
        plt.plot(xs, ys, marker=mk, linestyle=ls, markersize=6, linewidth=1.6, label=col)

    plt.xlabel("alpha")
    plt.ylabel("avg user data rate (Mbps)")
    plt.title(f"Avg user data rate (W={DEFAULT_W}, users={DEFAULT_USERS})")
    plt.grid(True, linestyle="--", alpha=0.3)
    if len(result.columns) > 6:
        plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout(rect=[0,0,0.78,1])
    else:
        plt.legend(loc='best')
        plt.tight_layout()

    # 6) 存檔到 <專案根>/results/
    out_png = os.path.join(RESULTS_DIR, f"avg_user_data_rate_W{DEFAULT_W}_users{DEFAULT_USERS}.png")
    out_csv = os.path.join(RESULTS_DIR, f"avg_user_data_rate_W{DEFAULT_W}_users{DEFAULT_USERS}_summary.csv")
    result.to_csv(out_csv)
    plt.savefig(out_png, dpi=OUT_DPI)
    print("\n已儲存彙總 CSV:", os.path.relpath(out_csv, PROJECT_ROOT))
    print("已儲存圖檔:", os.path.relpath(out_png, PROJECT_ROOT))

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
