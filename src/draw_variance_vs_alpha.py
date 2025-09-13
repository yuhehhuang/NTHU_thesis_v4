#!/usr/bin/env python3
# src/draw_variance_vs_alpha.py
# 只掃 <專案根>/results/ 下的 *_load_by_time.csv，計算 W=3、users=100（若檔名沒寫就放寬）
# 的「每個 time 的 load 方差」，再對 time 取平均，畫 x=alpha, y=variance 折線圖。
# dp 用實線，其它用虛線；每個方法 marker 不同。
#
# 另外輸出：
#   results/variance_W{W}_users{U}_vs_alpha_provenance_{ts}.csv   # 本次採用檔案
#   results/variance_W{W}_users{U}_vs_alpha_skipped_{ts}.csv      # 掃到但略過

import os, re, glob, math
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======== 可調參數 ========
DEFAULT_W = 3
DEFAULT_USERS = 100
STRICT_FILTER = True   # False=放寬：檔名沒寫 W 或 users 也接受；True=嚴格：一定要寫 W3 與 users100 才收
OUT_DPI = 150
# ==========================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))          # src/
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))    # 專案根
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")               # 只掃這裡
os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_alpha_from_name(name: str):
    name = name.replace('Α', 'alpha').replace('α', 'alpha')
    m = re.search(r'alpha[_\-]?([0-9]+(?:\.[0-9]+)?)', name, re.IGNORECASE)
    if m: return float(m.group(1))
    m2 = re.search(r'[_\-]a([0-9]+(?:\.[0-9]+)?)', name, re.IGNORECASE)
    if m2: return float(m2.group(1))
    return None

def parse_users_from_name(name: str):
    m = re.search(r'(?i)(?:^|[^A-Za-z0-9])users?(\d+)(?:$|[^A-Za-z0-9])', name)
    return int(m.group(1)) if m else None

def parse_w_from_name(name: str):
    m = re.search(r'(?i)(?:^|[^A-Za-z0-9])W(\d+)(?:$|[^A-Za-z0-9])', name)
    return int(m.group(1)) if m else None

def parse_method_from_name(name: str):
    base = os.path.basename(name)
    if "_W" in base: return base.split("_W")[0]
    if "_w" in base: return base.split("_w")[0]
    return base.split("_")[0] if "_" in base else "unknown"

def recursive_find_files(root_folder, patterns):
    found = []
    for pat in patterns:
        found += glob.glob(os.path.join(root_folder, "**", pat), recursive=True)
    # 去重並排序
    return sorted(list(dict.fromkeys(found)))

def build_style_list():
    markers = ['o','s','^','D','v','<','>','p','h','x','+','*','1','2','3','4']
    linestyles = ['-','--','-.',':']
    return [(mk, ls) for ls in linestyles for mk in markers]

def compute_variance_from_file(fpath: str):
    """回傳 (variance, stats_dict or error_dict)"""
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        return math.nan, {"error": f"read_csv failed: {e}"}

    has_time = "time" in df.columns
    has_load = "load" in df.columns
    if not (has_time and has_load):
        return math.nan, {
            "rows": len(df),
            "has_time": has_time,
            "has_load": has_load,
            "error": "missing columns (need time, load)"
        }

    rows = len(df)
    time_slots = df["time"].nunique()
    sats = df["sat"].nunique() if "sat" in df.columns else math.nan

    try:
        per_t = df.groupby("time")["load"].apply(
            lambda s: float(np.var(pd.to_numeric(s, errors="coerce"), ddof=0))
        )
        if len(per_t) == 0:
            return math.nan, {"rows": rows, "time_slots": time_slots, "sats": sats, "error": "empty per-time groups"}
        var_val = float(np.nanmean(per_t.values))
    except Exception as e:
        return math.nan, {"rows": rows, "time_slots": time_slots, "sats": sats, "error": f"variance calc failed: {e}"}

    return var_val, {"rows": rows, "time_slots": time_slots, "sats": sats, "has_time": True, "has_load": True}

def main():
    print("Search dir:", RESULTS_DIR)

    candidates = recursive_find_files(RESULTS_DIR, ["*_load_by_time.csv", "*load*by*time*.csv"])
    if not candidates:
        print("results/ 內找不到 *_load_by_time.csv。")
        return

    print(f"Found {len(candidates)} candidate files (showing up to 50):")
    for f in candidates[:50]:
        print(" -", os.path.relpath(f, PROJECT_ROOT))

    files = []
    for p in candidates:
        fname = os.path.basename(p)
        if STRICT_FILTER:
            w = parse_w_from_name(fname)          # 只看檔名
            u = parse_users_from_name(fname)      # 只看檔名
            if (w == DEFAULT_W) and (u == DEFAULT_USERS):
                files.append(p)
        else:
            w_file = parse_w_from_name(fname)
            u_file = parse_users_from_name(fname)
            norm = p.replace("\\", "/")
            w_path = parse_w_from_name(norm)
            u_path = parse_users_from_name(norm)
            w = w_file if w_file is not None else w_path
            u = u_file if u_file is not None else u_path
            cond_w = (w is None) or (w == DEFAULT_W)
            cond_u = (u is None) or (u == DEFAULT_USERS)
            if cond_w and cond_u:
                files.append(p)

    print(f"\nAfter filter (W={DEFAULT_W}, users={DEFAULT_USERS}, STRICT={STRICT_FILTER}): {len(files)} files")
    if not files:
        print("沒有符合條件的 *_load_by_time.csv。")
        return

    # === 蒐集＆去衝突（同 method, alpha → 取 mtime 最新） ===
    selected = {}     # key=(method_lower, alpha) -> row(dict) for provenance USED
    skipped_rows = [] # 略過清單（含被取代、缺欄位、沒 alpha 等）

    for fpath in files:
        fname   = os.path.basename(fpath)
        relpath = os.path.relpath(fpath, PROJECT_ROOT)
        method  = parse_method_from_name(fname)
        alpha   = parse_alpha_from_name(fname)
        if alpha is None:
            alpha = parse_alpha_from_name(fpath.replace("\\","/"))
        if alpha is None:
            skipped_rows.append({
                "status": "skipped",
                "reason": "alpha not parsed",
                "file": relpath,
                "method": method,
                "W": parse_w_from_name(fname),
                "users": parse_users_from_name(fname),
            })
            continue

        var_val, stats = compute_variance_from_file(fpath)
        if math.isnan(var_val):
            skipped_rows.append({
                "status": "skipped",
                "reason": stats.get("error", "variance is NaN"),
                "file": relpath,
                "method": method,
                "alpha": alpha,
                "W": parse_w_from_name(fname),
                "users": parse_users_from_name(fname),
                "rows": stats.get("rows"),
                "time_slots": stats.get("time_slots"),
                "sats": stats.get("sats"),
                "has_time": stats.get("has_time"),
                "has_load": stats.get("has_load"),
            })
            continue

        try:
            mtime_ts = os.path.getmtime(fpath)
            mtime_str = datetime.fromtimestamp(mtime_ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mtime_ts = -1.0
            mtime_str = ""

        key = (method.lower(), float(alpha))
        row = {
            "status": "used",
            "file": relpath,
            "method": method,
            "alpha": float(alpha),
            "W": parse_w_from_name(fname),
            "users": parse_users_from_name(fname),
            "variance": var_val,
            "rows": stats.get("rows"),
            "time_slots": stats.get("time_slots"),
            "sats": stats.get("sats"),
            "has_time": stats.get("has_time"),
            "has_load": stats.get("has_load"),
            "file_mtime": mtime_str,
            "_mtime_ts": mtime_ts,  # 內部排序用，不輸出
        }

        if key not in selected:
            selected[key] = row
        else:
            # 同 (method, alpha)：取 mtime 較新的為主
            prev = selected[key]
            if mtime_ts > prev["_mtime_ts"]:
                # 舊的被取代
                skipped_rows.append({
                    **{k: v for k, v in prev.items() if k != "_mtime_ts"},
                    "status": "skipped",
                    "reason": "superseded by newer file",
                })
                selected[key] = row
            else:
                # 目前這個較舊 → 略過
                skipped_rows.append({
                    **{k: v for k, v in row.items() if k != "_mtime_ts"},
                    "status": "skipped",
                    "reason": "older than selected file of same (method, alpha)",
                })

    if not selected:
        print("沒有可用的數據（可能所有檔案都缺 alpha 或資料空）。")
        if skipped_rows:
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            pd.DataFrame(skipped_rows).to_csv(
                os.path.join(RESULTS_DIR, f"variance_W{DEFAULT_W}_users{DEFAULT_USERS}_vs_alpha_skipped_{ts}.csv"),
                index=False, encoding="utf-8-sig"
            )
            print("已儲存 skipped 清單。")
        return

    # === 組表：index=alpha、columns=method ===
    # 只取最後選中的 used 檔案
    used_rows = [{k: v for k, v in r.items() if k != "_mtime_ts"} for r in selected.values()]
    methods = sorted(set(r["method"] for r in used_rows), key=lambda s: s.lower())
    alphas  = sorted(set(float(r["alpha"]) for r in used_rows))
    result = pd.DataFrame(index=alphas, columns=methods, dtype=float)
    for r in used_rows:
        result.at[float(r["alpha"]), r["method"]] = r["variance"]
    result.index.name = "alpha"

    print("\n彙總表 (alpha x method) – variance：")
    print(result)

    # === 畫圖 ===
    styles = build_style_list()
    plt.figure(figsize=(10,6))
    xs = result.index.values.astype(float)
    for i, col in enumerate(result.columns):
        ys = result[col].values.astype(float)
        if np.all(np.isnan(ys)):
            continue
        mk = styles[i % len(styles)][0]
        ls = '-' if 'dp' in col.lower() else '--'
        plt.plot(xs, ys, marker=mk, linestyle=ls, markersize=6, linewidth=1.6, label=col)

    plt.xlabel("alpha")
    plt.ylabel("variance of load (per-time avg)")
    plt.title(f"Load Variance vs alpha (W={DEFAULT_W}, users={DEFAULT_USERS}{'' if STRICT_FILTER else ' / relaxed'})")
    plt.grid(True, linestyle="--", alpha=0.3)
    if len(result.columns) > 6:
        plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout(rect=[0,0,0.78,1])
    else:
        plt.legend(loc='best')
        plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_png = os.path.join(RESULTS_DIR, f"variance_W{DEFAULT_W}_users{DEFAULT_USERS}_vs_alpha_{ts}.png")
    out_csv = os.path.join(RESULTS_DIR, f"variance_W{DEFAULT_W}_users{DEFAULT_USERS}_vs_alpha_summary.csv")
    result.to_csv(out_csv, encoding="utf-8-sig")
    plt.savefig(out_png, dpi=OUT_DPI)

    # === 溯源輸出 ===
    prov_csv = os.path.join(RESULTS_DIR, f"variance_W{DEFAULT_W}_users{DEFAULT_USERS}_vs_alpha_provenance_{ts}.csv")
    pd.DataFrame(used_rows).to_csv(prov_csv, index=False, encoding="utf-8-sig")

    if skipped_rows:
        skipped_csv = os.path.join(RESULTS_DIR, f"variance_W{DEFAULT_W}_users{DEFAULT_USERS}_vs_alpha_skipped_{ts}.csv")
        pd.DataFrame(skipped_rows).to_csv(skipped_csv, index=False, encoding="utf-8-sig")
        print("\n已儲存 skipped 清單：", os.path.relpath(skipped_csv, PROJECT_ROOT))

    print("\n已儲存彙總 CSV:", os.path.relpath(out_csv, PROJECT_ROOT))
    print("已儲存圖檔:", os.path.relpath(out_png, PROJECT_ROOT))
    print("已儲存 provenance 清單：", os.path.relpath(prov_csv, PROJECT_ROOT))

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
