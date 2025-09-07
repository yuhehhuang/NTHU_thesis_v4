#!/usr/bin/env python3
# src/draw_data_rates_vs_users.py
# 列出候選/過濾/實際讀到的 CSV（Terminal + trace 檔），並畫 x=users, y=avg user data rate

import os, re, glob, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======== 可調參數 ========
DEFAULT_W = 3
DEFAULT_ALPHA = 1.0
USERS_LIST = [100, 150, 200, 250, 300]
OUT_DPI = 150
ONLY_REAL = False  # 若檔名一定包含 "real"，設 True 更保險
# ==========================

# --- 基本路徑 ---
SCRIPT_PATH  = os.path.abspath(__file__)
SCRIPT_DIR   = os.path.dirname(SCRIPT_PATH)            # src/
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRACE_PATH   = os.path.join(RESULTS_DIR, "trace_draw_data_rates_vs_users.txt")
TRACE_FALLBACK = os.path.abspath("trace_fallback.txt")   # 寫不進 results 時的備援

# 讓 print 盡量即時輸出
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---- logger（Terminal + 檔案；檔案寫入失敗就用 fallback）----
def _write_file(path, msg):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return True
    except Exception as e:
        print(f"[log-error] write {path} failed: {e}", flush=True)
        return False

def log(msg=""):
    print(msg, flush=True)
    if not _write_file(TRACE_PATH, msg):
        _write_file(TRACE_FALLBACK, msg)

# ---- 開場標記：幫你確認到底跑到哪支檔 ----
def banner():
    # 先清掉舊 trace
    try:
        if os.path.exists(TRACE_PATH): os.remove(TRACE_PATH)
    except Exception:
        pass
    log("=== RUNNING draw_data_rates_vs_users.py ===")
    log(f"SCRIPT_PATH : {SCRIPT_PATH}")
    log(f"PYTHON      : {sys.version.split()[0]}")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"RESULTS_DIR : {RESULTS_DIR}")
    log(f"TRACE_PATH  : {TRACE_PATH}")
    log("==========================================")

# ---------- 小工具 ----------
def recursive_find_files(root_folder, patterns):
    found = []
    for pat in patterns:
        fullpat = os.path.join(root_folder, "**", pat)
        found.extend(glob.glob(fullpat, recursive=True))
    return sorted(list(dict.fromkeys(found)))

def parse_alpha_from_name(name: str):
    normalized = name.replace('Α', 'alpha').replace('α', 'alpha')
    m = re.search(r'alpha[_\-]?([0-9]+(?:\.[0-9]+)?)', normalized, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m2 = re.search(r'[_\-]a([0-9]+(?:\.[0-9]+)?)', name, re.IGNORECASE)
    if m2:
        return float(m2.group(1))
    return None

def find_rate_col(cols):
    cols_low = [c.lower() for c in cols]
    for i, c in enumerate(cols_low):
        if (re.search(r'avg.*data.*rate', c)
            or re.search(r'avg.*user.*rate', c)
            or re.search(r'avg.*rate', c)):
            return cols[i]
    numeric_candidates = []
    for c in cols:
        if c.lower() in ("time", "t", "alpha", "user_id", "userid", "users", "method"):
            continue
        numeric_candidates.append(c)
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    return None

def is_alpha_like_series(series, alpha_value):
    try:
        vals = pd.to_numeric(series, errors='coerce').dropna()
    except Exception:
        return False
    if vals.empty: return False
    if np.allclose(vals.values, np.full(len(vals), alpha_value), atol=1e-6): return True
    if vals.std() < 1e-6 and abs(vals.mean() - alpha_value) < 1e-6: return True
    if vals.nunique() == 1 and abs(vals.iloc[0] - alpha_value) < 1e-6: return True
    return False

def build_markers(): return ['o','s','^','D','v','<','>','p','h','x','+','*','1','2','3','4']

def filter_by_alpha(paths, alpha_target):
    """優先用檔名判斷 alpha；沒寫就讀 CSV 的 alpha 欄。回傳 (符合清單, 被跳過清單)。"""
    selected, skipped = [], []
    for p in paths:
        bn = os.path.basename(p)
        a = parse_alpha_from_name(bn)
        if a is not None:
            (selected if abs(a - alpha_target) < 1e-9 else skipped).append((p, a))
            continue
        # 檔名沒寫 alpha → 讀 CSV
        try:
            tmp = pd.read_csv(p, nrows=5)
            alpha_col = next((c for c in tmp.columns if c.lower() == 'alpha'), None)
            if alpha_col is not None:
                col = pd.read_csv(p, usecols=[alpha_col]).dropna()
                if not col.empty:
                    a = float(col.iloc[0, 0])
                    (selected if abs(a - alpha_target) < 1e-9 else skipped).append((p, a))
                else:
                    skipped.append((p, None))
            else:
                skipped.append((p, None))
        except Exception:
            skipped.append((p, None))
    return [p for p,_ in selected], skipped

# ---------- 主流程 ----------
def main():
    banner()

    W = DEFAULT_W
    ALPHA = DEFAULT_ALPHA
    USERS = USERS_LIST

    log(f"Target: W={W}, alpha={ALPHA}, users list={USERS}")

    per_users_files = {u: [] for u in USERS}
    for u in USERS:
        patterns = [
            f"avg_user_data_rate_W{W}_users{u}_*alpha{ALPHA:g}*.csv",
            f"avg_user_data_rate_*W{W}*_users{u}_*alpha{ALPHA:g}*.csv",
            f"avg_user_data_rate_W{W}_users{u}_*α{int(ALPHA)}*.csv",
            f"avg_user_data_rate_*W{W}*_users{u}_*α{int(ALPHA)}*.csv",
            f"avg_user_data_rate_W{W}_users{u}_*a{ALPHA:g}*.csv",
            f"avg_user_data_rate_*W{W}*_users{u}_*a{ALPHA:g}*.csv",
            f"avg_user_data_rate_W{W}_user{u}_*alpha{ALPHA:g}*.csv",
            f"avg_user_data_rate_*W{W}*_user{u}_*alpha{ALPHA:g}*.csv",
            f"avg_user_data_rate_W{W}_user{u}_*α{int(ALPHA)}*.csv",
            f"avg_user_data_rate_*W{W}*_user{u}_*α{int(ALPHA)}*.csv",
        ]
        found = recursive_find_files(PROJECT_ROOT, patterns)
        # 排除 summary、自身產出
        found = [p for p in found if "summary" not in os.path.basename(p).lower()]
        # 嚴格檢查：同時包含 W 與 users（避免奇怪命名順序）
        def _ok(bn: str) -> bool:
            w_ok = re.search(rf'(^|[_\-])W{W}([_\-]|users|user)', bn, re.IGNORECASE)
            u_ok = re.search(rf'(^|[_\-])users?{u}([_\-]|alpha|α|\.csv$)', bn, re.IGNORECASE)
            return bool(w_ok and u_ok)
        found = [p for p in found if _ok(os.path.basename(p))]

        if ONLY_REAL:
            found = [p for p in found if "real" in os.path.basename(p).lower()]

        # 列出候選清單（未過濾 α）
        log(f"\nUsers={u} -> candidates: {len(found)}")
        for p in found:
            log("  - " + os.path.relpath(p, PROJECT_ROOT))

        # 候選階段就過濾 α，並列出被跳過的
        alpha_ok, alpha_skipped = filter_by_alpha(found, ALPHA)
        for p, a in alpha_skipped:
            tag = "未知" if a is None else f"alpha={a:g}"
            log(f"  [skip α≠{ALPHA:g}] {os.path.relpath(p, PROJECT_ROOT)}  ({tag})")

        log(f"  [pick α={ALPHA:g}] ({len(alpha_ok)})")
        for p in alpha_ok:
            log("   -> " + os.path.relpath(p, PROJECT_ROOT))

        per_users_files[u] = alpha_ok

    chosen_files = {u: paths for u, paths in per_users_files.items() if paths}

    # 讀檔，彙總成 method x users 表
    data = {}   # data[method][users] = value
    for u, paths in chosen_files.items():
        per_file_method_avg = []
        for p in paths:
            rel = os.path.relpath(p, PROJECT_ROOT)
            log(f"[read] users={u} <- {rel}")

            try:
                df = pd.read_csv(p)
            except Exception as e:
                log(f"[error] users={u} {rel}: {e}")
                continue
            if df.empty:
                log(f"[warn] users={u} 空檔案：{rel}")
                continue

            method_col = next((c for c in ("method","Method","METHOD") if c in df.columns), None)
            rate_col = find_rate_col(list(df.columns))

            if method_col and rate_col:
                log(f"[use ] {rel}  method_col='{method_col}', rate_col='{rate_col}'")
                grp = df.groupby(method_col)[rate_col].mean()
                per_file_method_avg.append(grp)
            else:
                # 回退：選可能的數值欄平均（避開 alpha-like 欄）
                skip_names = {"alpha","a","w","users","user","time","t","id","index","method"}
                numeric_cols = []
                for c in df.columns:
                    if c.lower() in skip_names: continue
                    if re.search(r'\balpha\b', c, re.IGNORECASE) or 'α' in c: continue
                    ser = pd.to_numeric(df[c], errors='coerce').dropna()
                    if ser.empty: continue
                    if is_alpha_like_series(ser, ALPHA): continue
                    numeric_cols.append(c)
                if numeric_cols:
                    if len(numeric_cols) == 1:
                        ser = pd.to_numeric(df[numeric_cols[0]], errors='coerce')
                        val = float(ser.mean(skipna=True))
                    else:
                        vals_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                        val = float(np.nanmean(vals_df.to_numpy()))
                    log(f"[fallback] {rel} -> 用數值欄平均（cols={numeric_cols}）")
                    per_file_method_avg.append(pd.Series({"unknown_method": val}))

        if not per_file_method_avg:
            continue
        merged = pd.concat(per_file_method_avg, axis=1).mean(axis=1)
        for method, val in merged.items():
            data.setdefault(str(method), {})[u] = float(val)

    if not data:
        log("沒有可用數據可畫。")
        return

    # 組成 DataFrame：index=users, columns=methods
    methods = sorted(data.keys())
    users_sorted = sorted([u for u in USERS if any(u in d for d in [data[m] for m in methods])])
    result = pd.DataFrame(index=users_sorted, columns=methods, dtype=float)
    for m in methods:
        for u, v in data[m].items():
            result.at[u, m] = v
    result.index.name = "users"

    log("\n彙總表 (users x method):")
    log(str(result))

    # 作圖
    markers = build_markers()
    plt.figure(figsize=(10,6))
    xs = result.index.values
    for i, col in enumerate(result.columns):
        ys = result[col].values.astype(float)
        if np.all(np.isnan(ys)): continue
        mk = markers[i % len(markers)]
        ls = '-' if col.lower().startswith('dp') else '--'
        plt.plot(xs, ys, marker=mk, linestyle=ls, markersize=6, linewidth=1.8, label=col)

    plt.xlabel("Number of users")
    plt.ylabel("Average user data rate (Mbps)")
    plt.title(f"Avg user data rate vs users (W={DEFAULT_W}, alpha={DEFAULT_ALPHA:g})")
    plt.grid(True, linestyle="--", alpha=0.3)
    if len(result.columns) > 6:
        plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
        plt.tight_layout(rect=[0,0,0.78,1])
    else:
        plt.legend(loc='best')
        plt.tight_layout()

    # 存檔
    out_csv = os.path.join(RESULTS_DIR, f"avg_user_data_rate_vs_users_W{DEFAULT_W}_alpha{DEFAULT_ALPHA:g}.csv")
    out_png = os.path.join(RESULTS_DIR, f"avg_user_data_rate_vs_users_W{DEFAULT_W}_alpha{DEFAULT_ALPHA:g}.png")
    log(f"\n[summary] trace file   : {os.path.relpath(TRACE_PATH, PROJECT_ROOT)}")
    log(f"[summary] fallback log: {TRACE_FALLBACK}")
    result.to_csv(out_csv)
    plt.savefig(out_png, dpi=OUT_DPI)
    log(f"已儲存彙總 CSV: {os.path.relpath(out_csv, PROJECT_ROOT)}")
    log(f"已儲存圖檔: {os.path.relpath(out_png, PROJECT_ROOT)}")

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
