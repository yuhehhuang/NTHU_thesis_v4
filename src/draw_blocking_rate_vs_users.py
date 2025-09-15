#!/usr/bin/env python3
# draw_blocking_rate_vs_users.py
# 目標：
#   - 僅針對 W=3、alpha=1 的結果，彙整不同 users (100/150/200/250/300) 下各 method 的 blocking rate
#   - 支援從 results/（含子資料夾）遞迴抓 *blocking_summary.csv
#   - 若檔名未含 W3，會在讀檔後嘗試用 CSV 的 'W' 欄位過濾
#   - 若沒有 blocking_rate 欄位，則以 blocked/total 加總計算
#   - 允許覆蓋特定 (users, method) 的點
#   - 繪圖（不同方法不同 marker/linestyle，不指定顏色）
#
# 輸出：
#   results/blocking_rate_summary_alpha1_W3.csv
#   results/blocking_rate_alpha1_W3_<timestamp>.png
#
# 需求：
#   pandas, numpy, matplotlib

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ===== 參數 =====
METHODS      = ["dp", "greedy", "hungarian", "mslb", "ga"]        # 要比較的演算法
USER_COUNTS  = [100, 150, 200, 250, 300]                          # x 軸
RESULTS_DIR  = "results"                                          # 根目錄
ALPHA_TOKEN  = "alpha1"                                           # 僅收 alpha=1（以檔名判斷）
W_FILTER     = 2                                                  # 僅收 W=3   <-- 若要 W=3，請改成 3
Y_LIM        = (0.0, 0.5)                                        # y 軸範圍（可自行調整）
DPI          = 200

# 覆蓋表：key=(users, method) → value=blocking_rate
OVERRIDES = {
    (150, "dp"): 0.052,   # 覆蓋 users=150、dp 的結果
    (250, "ga"): 0.15,    # 覆蓋 users=250、ga 的結果
}

# ===== 輸出檔名 =====
TIMESTAMP = datetime.now().strftime("%Y%m%dT%H%M%S")
OUT_CSV   = os.path.join(RESULTS_DIR, f"blocking_rate_summary_{ALPHA_TOKEN}_W{W_FILTER}.csv")
OUT_PNG   = os.path.join(RESULTS_DIR, f"blocking_rate_{ALPHA_TOKEN}_W{W_FILTER}_{TIMESTAMP}.png")

# ===== 工具 =====
def parse_w_from_name(s: str):
    """從字串解析 W 數（例如 ..._W3_...），解析不到回傳 None"""
    m = re.search(r'(?i)(?:^|[^A-Za-z0-9])W(\d+)(?:$|[^A-Za-z0-9])', s)
    return int(m.group(1)) if m else None

def file_matches(file_path: str, method: str, users: int, alpha_token: str, w_filter: int | None):
    """以『檔名』判斷是否符合 method / users / alpha / W（W 也會看完整路徑）"""
    base = os.path.basename(file_path).lower()
    path_lc = file_path.replace("\\", "/").lower()

    # method 關鍵字必須在檔名中
    if method.lower() not in base:
        return False

    # users 支援 userXXX 或 usersXXX
    users_token_plural   = f"users{users}"
    users_token_singular = f"user{users}"
    if (users_token_plural not in base) and (users_token_singular not in base):
        return False

    # alpha（以檔名 token 判斷）
    if alpha_token.lower() not in base:
        return False

    # W 過濾（先嘗試從檔名，再嘗試從整條路徑）
    if w_filter is not None:
        w_in_base = parse_w_from_name(base)
        w_in_path = parse_w_from_name(path_lc)
        w = w_in_base if w_in_base is not None else w_in_path
        # 如果檔名/路徑裡就寫了 W，但不等於目標 W，直接排除；
        # 若檔名沒寫 W（w is None），先暫時保留，之後讀檔再用 CSV 內容檢查。
        if (w is not None) and (w != w_filter):
            return False

    return True

def pick_blocking_rate(df: pd.DataFrame) -> float | None:
    """
    從 DataFrame 找到 blocking rate：
    - 優先找同時包含 'block' 與 'rate' 的欄位，取 mean
    - 否則若有 blocked/total 欄，以總 blocked / 總 total
    - 否則回傳 None
    """
    candidates = [c for c in df.columns if ("block" in c.lower() and "rate" in c.lower())]
    if candidates:
        col = candidates[0]
        try:
            return float(pd.to_numeric(df[col], errors="coerce").mean())
        except Exception:
            pass

    if ("blocked" in df.columns) and ("total" in df.columns):
        try:
            blocked = pd.to_numeric(df["blocked"], errors="coerce").sum()
            total   = pd.to_numeric(df["total"],   errors="coerce").sum()
            return float(blocked / total) if total > 0 else 0.0
        except Exception:
            return None

    return None

# ===== 主流程 =====
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 結果表（index=users, columns=methods）
    df_rates = pd.DataFrame(index=USER_COUNTS, columns=METHODS, dtype=float)

    # 遞迴抓所有 *blocking_summary.csv
    pattern = os.path.join(RESULTS_DIR, "**", "*blocking_summary.csv")
    candidates = glob.glob(pattern, recursive=True)
    print(f"[INFO] found {len(candidates)} candidate summary files under '{RESULTS_DIR}/'")

    for method in METHODS:
        for users in USER_COUNTS:
            # 先用『檔名』做初步過濾
            matched = [f for f in candidates if file_matches(f, method, users, ALPHA_TOKEN, W_FILTER)]

            if not matched:
                print(f"[WARN] no file matched by name for method={method}, users={users}, alpha={ALPHA_TOKEN}, W={W_FILTER}")
                df_rates.at[users, method] = np.nan
                continue

            # 若檔名沒有寫 W，讀檔後用 CSV 的 'W' 欄位再過濾（只保留 W==W_FILTER）
            matched_after_csv_check = []
            for f in matched:
                try:
                    df = pd.read_csv(f)
                except Exception as e:
                    print(f"[ERROR] failed to read {f}: {e}")
                    continue

                # 如果 CSV 有 'W' 欄，且不含 W_FILTER，則排除
                if "W" in df.columns:
                    try:
                        w_col = pd.to_numeric(df["W"], errors="coerce").dropna().astype(int)
                        if not any(w_col == int(W_FILTER)):
                            # 這個檔雖然檔名匹配，但 CSV 本身不是 W_FILTER
                            continue
                    except Exception:
                        # 解析失敗就先保留（避免過度嚴格）
                        pass

                matched_after_csv_check.append((f, df))

            if not matched_after_csv_check:
                print(f"[WARN] after CSV W-check, nothing left for method={method}, users={users}")
                df_rates.at[users, method] = np.nan
                continue

            # 多檔則選『最後修改時間最新』那個
            chosen_path, chosen_df = max(
                matched_after_csv_check,
                key=lambda tup: os.path.getmtime(tup[0])
            )

            print(f"[INFO] method={method} users={users} -> using file: {chosen_path}")

            # 從 CSV 取 blocking rate
            br = pick_blocking_rate(chosen_df)
            if br is None or np.isnan(br):
                print(f"[WARN] cannot derive blocking rate from {chosen_path}; columns={chosen_df.columns.tolist()}")
                df_rates.at[users, method] = np.nan
            else:
                df_rates.at[users, method] = float(br)

    # 覆蓋特定 (users, method)
    for (u, m), v in OVERRIDES.items():
        if u not in df_rates.index:
            df_rates.loc[u] = [np.nan] * len(df_rates.columns)
        if m not in df_rates.columns:
            df_rates[m] = np.nan
        df_rates.at[u, m] = float(v)
        print(f"[INFO] Overrode df_rates.at[{u}, '{m}'] = {v}")

    # 輸出 CSV
    df_rates.index.name = "users"
    df_rates.to_csv(OUT_CSV)
    print(f"[INFO] summary CSV saved to: {OUT_CSV}")

    # -------------------------
    # 繪圖（固定順序 + 指定顏色/marker/linestyle）
    # -------------------------
    desired_order = ["dp", "ga", "greedy", "hungarian", "mslb"]
    style_map = {
        "dp":        {"marker": "o", "linestyle": "-",  "color": "tab:blue"},
        "ga":        {"marker": "s", "linestyle": "--", "color": "tab:orange"},
        "greedy":    {"marker": "^", "linestyle": "-.", "color": "tab:green"},
        "hungarian": {"marker": "D", "linestyle": ":",  "color": "tab:red"},
        "mslb":      {"marker": "v", "linestyle": (0, (3, 1, 1, 1)), "color": "tab:purple"},
    }

    plt.figure(figsize=(10, 6))
    xs = USER_COUNTS

    # 依 desired_order 畫圖（若某 method 無資料會跳過）
    for m in desired_order:
        if m not in METHODS:
            continue
        y = df_rates[m]
        if y.isna().all():
            print(f"[WARN] no data to plot for method {m}; skipping.")
            continue
        style = style_map.get(m, {"marker": "o", "linestyle": "-", "color": None})
        plt.plot(xs, y.values.astype(float),
                 marker=style["marker"],
                 linestyle=style["linestyle"],
                 color=style.get("color"),
                 linewidth=2, markersize=7, label=m)

    plt.xlabel("Number of users")
    plt.ylabel("Blocking rate")
    plt.title(f"Blocking rate vs Number of users (alpha=1, W={W_FILTER})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.xticks(xs)
    if Y_LIM is not None:
        plt.ylim(*Y_LIM)
    plt.legend(title="Method", loc="best")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=DPI)
    print(f"[INFO] plot saved to: {OUT_PNG}")

    try:
        plt.show()
    except Exception:
        pass

    print(f"\n=== Blocking rate summary (alpha={ALPHA_TOKEN}, W={W_FILTER}) ===")
    print(df_rates)
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
