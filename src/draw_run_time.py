#!/usr/bin/env python3
# src/draw_run_time.py
# 從 results/method_timing.csv 讀取 W=3、users={100,150,200,250,300} 的各方法執行時間
# 畫 X=users、Y=seconds（每方法一條線），並輸出彙總 CSV / PNG 與樣本數統計。

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---- 預設參數 ----
DEFAULT_W = 3
DEFAULT_USERS = [100, 150, 200, 250, 300]
DEFAULT_FILE = None  # 會在程式內設成 <專案根>/results/method_timing.csv
DEFAULT_AGG = "min"  # 可選: median / mean / min / max
OUT_DPI = 180

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))          # src/
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))    # 專案根
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

STYLE_MAP = {
    "dp":        {"marker": "o", "linestyle": "-"},
    "greedy":    {"marker": "s", "linestyle": "--"},
    "hungarian": {"marker": "^", "linestyle": "-."},
    "mslb":      {"marker": "D", "linestyle": ":"},
    "ga":        {"marker": "x", "linestyle": (0, (3, 1, 1, 1))},
}

def pick_elapsed_column(df: pd.DataFrame) -> str:
    """挑出代表執行時間的欄位（優先 elapsed_sec），找不到就報錯。"""
    candidates = ["elapsed_sec", "runtime_sec", "run_time", "elapsed", "seconds", "duration"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"找不到執行時間欄位（嘗試過: {candidates}）")

def aggregate_series(s: pd.Series, how: str) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return np.nan
    how = how.lower()
    if how == "median":
        return float(s.median())
    if how == "mean":
        return float(s.mean())
    if how == "min":
        return float(s.min())
    if how == "max":
        return float(s.max())
    raise ValueError("agg 只能是 median/mean/min/max")

def main():
    parser = argparse.ArgumentParser(description="Draw runtime vs users from method_timins.csv")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help="輸入 CSV（預設 results/method_timings.csv）")
    parser.add_argument("--w", type=int, default=DEFAULT_W, help="目標 W（預設 3）")
    parser.add_argument("--users", default="100,150,200,250,300",
                        help="逗號分隔的 users 清單，預設 100,150,200,250,300")
    parser.add_argument("--alpha", default="any",
                        help="指定 alpha（數值），或 'any' 代表全部聚合")
    parser.add_argument("--agg", default=DEFAULT_AGG, choices=["median","mean","min","max"],
                        help="聚合方式（預設 median）")
    args = parser.parse_args()

    infile = args.file or os.path.join(RESULTS_DIR, "method_timings.csv")
    users_list = [int(x) for x in str(args.users).split(",") if x.strip()]
    target_w = int(args.w)
    alpha_filter = None if str(args.alpha).lower() == "any" else float(args.alpha)

    if not os.path.isfile(infile):
        raise FileNotFoundError(f"找不到輸入檔：{infile}")

    df = pd.read_csv(infile)
    required = {"method", "W", "elapsed_sec", "num_users"}
    if not required.issubset(df.columns):
        # 容錯：若沒有 elapsed_sec，就挑其他欄位
        elapsed_col = pick_elapsed_column(df)
    else:
        elapsed_col = "elapsed_sec"

    # 正規化欄位
    if "method" not in df.columns:
        raise ValueError("CSV 缺少 'method' 欄位")
    if "W" not in df.columns:
        raise ValueError("CSV 缺少 'W' 欄位")
    if "num_users" not in df.columns:
        raise ValueError("CSV 缺少 'num_users' 欄位")

    # 過濾 W 與 users
    mask = (df["W"].astype(int) == target_w) & (df["num_users"].astype(int).isin(users_list))
    if alpha_filter is not None and "alpha" in df.columns:
        mask &= (pd.to_numeric(df["alpha"], errors="coerce") == alpha_filter)

    df_sel = df.loc[mask].copy()
    if df_sel.empty:
        raise ValueError(f"沒有找到符合條件的資料：W={target_w}, users∈{users_list}"
                         + (f", alpha={alpha_filter}" if alpha_filter is not None else ""))

    # 轉數值
    df_sel["num_users"] = pd.to_numeric(df_sel["num_users"], errors="coerce").astype("Int64")
    df_sel[elapsed_col] = pd.to_numeric(df_sel[elapsed_col], errors="coerce")

    # 以 (method, num_users) 聚合
    grp = df_sel.groupby(["method", "num_users"])[elapsed_col].apply(lambda s: aggregate_series(s, args.agg))
    # 同時算樣本數（觀察用）
    cnt = df_sel.groupby(["method", "num_users"])[elapsed_col].count().rename("n_samples")

    # 組寬表
    result = grp.unstack(0)  # index=num_users, columns=method
    result = result.reindex(users_list)  # 排序 users 軸
    result.index.name = "users"

    # 樣本數對照
    count_tbl = cnt.unstack(0).reindex(users_list)
    count_tbl.index.name = "users"

    print("\n彙總（單位：秒）:")
    print(result)

    # 存檔
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_csv = os.path.join(RESULTS_DIR, f"run_time_W{target_w}_vs_users_summary.csv")
    out_cnt = os.path.join(RESULTS_DIR, f"run_time_W{target_w}_vs_users_counts.csv")
    out_png = os.path.join(RESULTS_DIR, f"run_time_W{target_w}_vs_users_{ts}.png")
    result.to_csv(out_csv)
    count_tbl.to_csv(out_cnt)

    # 畫圖
    plt.figure(figsize=(10, 6))
    xs = np.array(users_list, dtype=int)
    for method in result.columns:
        ys = result[method].values.astype(float)
        if np.all(np.isnan(ys)):
            continue
        style = STYLE_MAP.get(str(method).lower(), {"marker": "o", "linestyle": "-"})
        plt.plot(xs, ys,
                 marker=style["marker"],
                 linestyle=style["linestyle"] if str(method).lower() != "dp" else "-",
                 linewidth=2, markersize=7, label=str(method))

    title_alpha = "" if alpha_filter is None else f", alpha={alpha_filter:g}"
    plt.xlabel("Number of users")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime vs Users ")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xticks(xs)
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(out_png, dpi=OUT_DPI)

    print("\n已儲存：")
    print("  彙總CSV  :", os.path.relpath(out_csv, PROJECT_ROOT))
    print("  樣本數CSV:", os.path.relpath(out_cnt, PROJECT_ROOT))
    print("  圖檔PNG  :", os.path.relpath(out_png, PROJECT_ROOT))

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
