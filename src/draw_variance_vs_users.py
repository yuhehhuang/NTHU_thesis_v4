#!/usr/bin/env python3
# draw_variance_vs_users.py
# 目的：
#   固定 W=3、alpha=1，從 results/ 讀取各方法在不同 users 數的 *_load_by_time.csv，
#   計算「每個 time 的 load(跨 sat) 方差」再對 time 取平均，繪製 x=users, y=variance 折線圖。
#
# 使用：
#   python src/draw_variance_vs_users.py
#
# 產出：
#   results/variance_W3_alpha1_vs_users_summary.csv
#   results/variance_W3_alpha1_vs_users_<timestamp>.png
#   results/variance_W3_alpha1_vs_users_provenance_<timestamp>.csv      <-- 新增
#   results/variance_W3_alpha1_vs_users_skipped_<timestamp>.csv         <-- 新增（如有）

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ======== 可調參數 ========
RESULTS_DIR   = "results"          # 只掃這個資料夾
TARGET_W      = 2                  # 固定 W=3
TARGET_ALPHA  = 1                  # 固定 alpha=1
METHODS       = ["dp", "greedy", "hungarian", "mslb", "ga"]  # 要畫的演算法（會過濾）
USER_COUNTS   = [100, 150, 200, 250, 300]  # 若想自動由檔名抓，設為 None
STRICT_FILTER = True               # True：檔名需明確含 W3、alpha1、usersX；False：檔名沒寫也接受（視為預設）
OUT_DPI       = 180                # 圖片解析度
AUDIT_DETAILS = True               # 是否輸出本次使用到/略過的檔案清單
# ==========================

# ---------- 小工具：從檔名/路徑解析 method, W, users, alpha ----------
def parse_alpha_from_name(name: str):
    # 支援 α/alpha 寫法
    normalized = name.replace('Α', 'alpha').replace('α', 'alpha')
    m = re.search(r'(?i)alpha[_\-]?([0-9]+(?:\.[0-9]+)?)', normalized)
    if m:
        return float(m.group(1))
    m2 = re.search(r'(?i)(?:^|[^A-Za-z0-9])a([0-9]+(?:\.[0-9]+)?)(?:$|[^A-Za-z0-9])', normalized)
    return float(m2.group(1)) if m2 else None

def parse_users_from_name(name: str):
    m = re.search(r'(?i)(?:^|[^A-Za-z0-9])users?(\d+)(?:$|[^A-Za-z0-9])', name)
    return int(m.group(1)) if m else None

def parse_w_from_name(name: str):
    m = re.search(r'(?i)(?:^|[^A-Za-z0-9])W(\d+)(?:$|[^A-Za-z0-9])', name)
    return int(m.group(1)) if m else None

def parse_method_from_name(name: str):
    base = os.path.basename(name)
    # 常見命名：<method>_W3_users100_alpha1_load_by_time.csv
    if "_W" in base or "_w" in base:
        # 避免 split 後沒有第二段的極端命名，保守處理
        try:
            return base.split("_W")[0].split("_w")[0]
        except Exception:
            pass
    # 後備：用第一段
    return base.split("_")[0] if "_" in base else base

# ---------- 計算一個檔案的「時間平均方差」 ----------
def compute_variance_from_file(fpath: str) -> (float, dict):
    """
    回傳 (variance, stats_dict)
    stats_dict 會包含 rows, time_slots, sats(若有 sat 欄), has_time, has_load
    """
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        return np.nan, {"error": f"read_csv failed: {e}"}

    has_time = "time" in df.columns
    has_load = "load" in df.columns
    if not (has_time and has_load):
        return np.nan, {
            "rows": len(df),
            "has_time": has_time,
            "has_load": has_load,
            "error": "missing columns"
        }

    rows = len(df)
    time_slots = df["time"].nunique()
    sats = df["sat"].nunique() if "sat" in df.columns else np.nan

    # 對每個 time，計算「跨 sat 的 load 方差」（母體方差 ddof=0）
    try:
        per_t = df.groupby("time")["load"].apply(
            lambda s: float(np.var(pd.to_numeric(s, errors="coerce"), ddof=0))
        )
        if len(per_t) == 0:
            return np.nan, {"rows": rows, "time_slots": time_slots, "sats": sats, "error": "empty per-time groups"}
        var_val = float(np.nanmean(per_t.values))
    except Exception as e:
        return np.nan, {"rows": rows, "time_slots": time_slots, "sats": sats, "error": f"variance calc failed: {e}"}

    return var_val, {"rows": rows, "time_slots": time_slots, "sats": sats, "has_time": True, "has_load": True}

def main():
    root = RESULTS_DIR
    if not os.path.isdir(root):
        print(f"[ERROR] results folder not found: {root}")
        return

    print(f"Scan dir: {os.path.abspath(root)}")
    # 只抓 *_load_by_time.csv
    candidates = glob.glob(os.path.join(root, "**", "*_load_by_time.csv"), recursive=True)
    print(f"Found {len(candidates)} load_by_time files (showing up to 40):")
    for p in candidates[:40]:
        print(" -", os.path.relpath(p, start=root))

    files = []
    for p in candidates:
        base = os.path.basename(p)
        # 嚴格模式：只看檔名本身是否含 W、alpha、users
        if STRICT_FILTER:
            w  = parse_w_from_name(base)
            a  = parse_alpha_from_name(base)
            u  = parse_users_from_name(base)
            if (w == TARGET_W) and (a == TARGET_ALPHA) and (u is not None):
                files.append(p)
        else:
            # 放寬：檔名抓不到時，從整條路徑補抓
            w_file = parse_w_from_name(base)
            a_file = parse_alpha_from_name(base)
            u_file = parse_users_from_name(base)

            norm_path = p.replace("\\","/")
            w_path = parse_w_from_name(norm_path)
            a_path = parse_alpha_from_name(norm_path)
            u_path = parse_users_from_name(norm_path)

            w = w_file if w_file is not None else w_path
            a = a_file if a_file is not None else a_path
            u = u_file if u_file is not None else u_path

            cond_w = (w is None) or (w == TARGET_W)
            cond_a = (a is None) or (a == TARGET_ALPHA)
            cond_u = (u is not None)  # 至少要知道 users 是多少
            if cond_w and cond_a and cond_u:
                files.append(p)

    print(f"\nAfter filter (W={TARGET_W}, alpha={TARGET_ALPHA}, STRICT={STRICT_FILTER}): {len(files)} files")
    if not files:
        print("沒有符合條件的 *_load_by_time.csv。")
        return

    # 收集資料與溯源
    data = {}
    users_seen = set()
    audit_rows = []   # 採用
    skipped_rows = [] # 掃到但略過

    for fpath in files:
        fname  = os.path.basename(fpath)
        rel    = os.path.relpath(fpath, start=root)
        method = parse_method_from_name(fname).lower()
        if METHODS and method not in [m.lower() for m in METHODS]:
            skipped_rows.append({
                "status": "skipped",
                "reason": f"method '{method}' not in METHODS filter",
                "file": rel,
                "method": method,
                "users": parse_users_from_name(fname),
                "W": parse_w_from_name(fname),
                "alpha": parse_alpha_from_name(fname),
            })
            continue

        # users
        users = parse_users_from_name(fname)
        if users is None and not STRICT_FILTER:
            users = parse_users_from_name(fpath.replace("\\","/"))
        if users is None:
            skipped_rows.append({
                "status": "skipped",
                "reason": "users not parsed from name/path",
                "file": rel,
                "method": method,
                "W": parse_w_from_name(fname),
                "alpha": parse_alpha_from_name(fname),
            })
            continue

        # 計算
        var_val, stats = compute_variance_from_file(fpath)
        if np.isnan(var_val):
            skipped_rows.append({
                "status": "skipped",
                "reason": stats.get("error", "variance is NaN"),
                "file": rel,
                "method": method,
                "users": users,
                "W": parse_w_from_name(fname),
                "alpha": parse_alpha_from_name(fname),
                "rows": stats.get("rows"),
                "time_slots": stats.get("time_slots"),
                "sats": stats.get("sats"),
                "has_time": stats.get("has_time"),
                "has_load": stats.get("has_load"),
            })
            continue

        # 成功採用
        data.setdefault(method, {})[users] = var_val
        users_seen.add(users)

        # 檔案 mtime
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mtime = ""

        audit_rows.append({
            "status": "used",
            "file": rel,
            "method": method,
            "users": users,
            "W": parse_w_from_name(fname),
            "alpha": parse_alpha_from_name(fname),
            "variance": var_val,
            "rows": stats.get("rows"),
            "time_slots": stats.get("time_slots"),
            "sats": stats.get("sats"),
            "has_time": stats.get("has_time"),
            "has_load": stats.get("has_load"),
            "file_mtime": mtime,
        })

    if not data:
        print("沒有可用的數據（可能所有檔案都缺 users 或資料空）。")
        # 仍然輸出 skipped 方便排查
        if AUDIT_DETAILS and skipped_rows:
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            os.makedirs(RESULTS_DIR, exist_ok=True)
            pd.DataFrame(skipped_rows).to_csv(
                os.path.join(RESULTS_DIR, f"variance_W{TARGET_W}_alpha{TARGET_ALPHA}_vs_users_skipped_{ts}.csv"),
                index=False, encoding="utf-8-sig"
            )
            print("已儲存 skipped 清單。")
        return

    # users 軸次序：若有指定 USER_COUNTS 就照指定，否則使用掃到的排序
    if USER_COUNTS:
        users_axis = USER_COUNTS
    else:
        users_axis = sorted(users_seen)

    # 組 DataFrame：index=users, columns=method
    methods_sorted = [m for m in METHODS if m.lower() in data.keys()]
    if not methods_sorted:
        methods_sorted = sorted(data.keys())

    result = pd.DataFrame(index=users_axis, columns=methods_sorted, dtype=float)
    for m in methods_sorted:
        for u, v in data[m].items():
            if u in result.index:
                result.at[u, m] = v
    result.index.name = "users"

    print("\n彙總表 (users x method) – variance：")
    print(result)

    # 畫圖：不同方法不同 marker/linestyle；dp 用實線，其餘照 style_map
    style_map = {
        "dp":        {"marker": "o", "linestyle": "-"},
        "greedy":    {"marker": "s", "linestyle": "--"},
        "hungarian": {"marker": "^", "linestyle": "-."},
        "mslb":      {"marker": "D", "linestyle": ":"},
        "ga":        {"marker": "x", "linestyle": (0, (3, 1, 1, 1))},
    }

    plt.figure(figsize=(10,6))
    xs = result.index.values.astype(int)
    for col in result.columns:
        ys = result[col].values.astype(float)
        if np.all(np.isnan(ys)):
            continue
        style = style_map.get(col.lower(), {"marker": "o", "linestyle": "-"})
        plt.plot(xs, ys,
                 marker=style["marker"],
                 linestyle=style["linestyle"] if col.lower() != "dp" else "-",
                 linewidth=2, markersize=7, label=col)

    plt.xlabel("Number of users")
    plt.ylabel("Variance of load (per-time average)")
    plt.title(f"Load Variance vs Users (W={TARGET_W}, alpha={TARGET_ALPHA})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xticks(xs)
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    out_csv = os.path.join(RESULTS_DIR, f"variance_W{TARGET_W}_alpha{TARGET_ALPHA}_vs_users_summary.csv")
    out_png = os.path.join(RESULTS_DIR, f"variance_W{TARGET_W}_alpha{TARGET_ALPHA}_vs_users_{ts}.png")
    result.to_csv(out_csv, encoding="utf-8-sig")
    plt.legend(title="Method")
    plt.savefig(out_png, dpi=OUT_DPI)

    print("\n已儲存彙總 CSV：", out_csv)
    print("已儲存圖檔：", out_png)

    # === 新增：輸出溯源（使用與略過） ===
    if AUDIT_DETAILS:
        if audit_rows:
            audit_csv = os.path.join(RESULTS_DIR, f"variance_W{TARGET_W}_alpha{TARGET_ALPHA}_vs_users_provenance_{ts}.csv")
            pd.DataFrame(audit_rows).to_csv(audit_csv, index=False, encoding="utf-8-sig")
            print("已儲存 provenance 清單：", audit_csv)
        if skipped_rows:
            skipped_csv = os.path.join(RESULTS_DIR, f"variance_W{TARGET_W}_alpha{TARGET_ALPHA}_vs_users_skipped_{ts}.csv")
            pd.DataFrame(skipped_rows).to_csv(skipped_csv, index=False, encoding="utf-8-sig")
            print("已儲存 skipped 清單：", skipped_csv)

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
