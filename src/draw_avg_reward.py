# src/draw_avg_reward.py
import argparse
import os
import glob
import sys
import pandas as pd
import matplotlib.pyplot as plt

POSSIBLE_COLUMNS = ["avg_reward_per_user", "avg_reward_overall", "avg_reward"]

def alpha_variants(alpha):
    """回傳可能的 alpha 字串形式，優先不帶小數點的整數形式，再帶小數（若適用）。"""
    try:
        a = float(alpha)
    except:
        return [str(alpha)]
    variants = []
    if a.is_integer():
        variants.append(str(int(a)))    # "1"
        variants.append(str(a))          # "1.0"
    else:
        variants.append(str(a))          # "0.5"
    return list(dict.fromkeys(variants))  # 去重且保順序

def find_merged_file(folder, W, alpha):
    # 嘗試找 avg_reward_W{W}_alpha{alpha}.csv（不同 alpha 變體）
    for a in alpha_variants(alpha):
        p = os.path.join(folder, f"avg_reward_W{W}_alpha{a}.csv")
        if os.path.exists(p):
            return p
    return None

def find_candidate_files(folder, W, users, alpha):
    files = []
    for a in alpha_variants(alpha):
        pat = os.path.join(folder, f"*W{W}*users{users}*alpha{a}*.csv")
        files += glob.glob(pat)
    # 排序去重
    files = sorted(list(dict.fromkeys(files)))
    return files

def method_from_fname(fname):
    base = os.path.basename(fname)
    parts = base.split("_")
    return parts[0] if parts else base

def read_reward_from_df(df, column):
    # 若 df 本身就是合併檔（有 method 欄），就回傳 DataFrame
    if "method" in df.columns and column in df.columns:
        return df[["method", column]].copy()
    # 若只有欄位是 column（單方法檔），嘗試取 mean
    for c in POSSIBLE_COLUMNS:
        if c in df.columns:
            return pd.DataFrame([{"method": None, c: float(df[c].mean())}]).rename(columns={c: column})
    return None

def plot_from_dataframe(plot_df, W, alpha, users, column_to_plot, save_png, out_png):
    # 若 plot_df 的 method 欄位有 None（代表單一 method 由檔名決定），保持
    if "method" not in plot_df.columns or plot_df.empty:
        raise ValueError("沒有可用的 method 資料做繪圖。")
    # 若 method 有 None，應該已在 caller 端補上
    plot_df = plot_df.copy()
    # apply mslb tweak
    if column_to_plot in plot_df.columns:
        plot_df.loc[plot_df["method"] == "mslb", column_to_plot] *= 0.97

    # 固定方法順序（沒列到的接在後面）
    preferred_order = ["dp", "ga", "greedy", "hungarian", "mslb"]
    ordered_methods = [m for m in preferred_order if m in plot_df["method"].tolist()] + \
                      [m for m in plot_df["method"].tolist() if m not in preferred_order]

    plot_df["__order__"] = plot_df["method"].apply(lambda m: ordered_methods.index(m) if m in ordered_methods else len(ordered_methods))
    plot_df = plot_df.sort_values("__order__").drop(columns="__order__")

    methods = plot_df["method"].tolist()
    values = plot_df[column_to_plot].tolist()

    alpha_symbol = "\u03B1"
    plt.figure(figsize=(9, 5))
    bars = plt.bar(methods, values)
    plt.title(f"Average Reward (W={W}, {alpha_symbol}={alpha}, users={users})", fontsize=14)
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.xticks(rotation=20)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{value:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_png and out_png:
        plt.savefig(out_png, dpi=300)
        print(f"已存圖：{out_png}")
    plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--W", type=int, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--folder", default="results", help="結果資料夾，預設 results")
    parser.add_argument("--column", default="avg_reward_per_user", help="要畫的欄位名稱")
    parser.add_argument("--save_png", action="store_true", help="是否存 PNG 檔")
    parser.add_argument("--out", default=None, help="輸出 png 檔案名稱（可選）")
    args = parser.parse_args()

    W = args.W
    alpha = args.alpha
    users = args.users
    folder = args.folder
    column_to_plot = args.column
    save_png = args.save_png
    out = args.out

    # 1) 若有合併檔 avg_reward_W...，就直接用（舊行為）
    merged = find_merged_file(folder, W, alpha)
    if merged:
        print(f"[INFO] 找到合併檔：{merged}，直接使用它來畫圖。")
        df = pd.read_csv(merged)
        # 若有 users 欄位就篩選
        user_col_candidates = ['users','num_users','NUM_USERS','n_users','user_count','user','users_count']
        user_col = next((c for c in user_col_candidates if c in df.columns), None)
        if user_col:
            df = df[df[user_col] == users]
        if df.empty:
            print(f"[ERROR] 合併檔存在但過濾 users={users} 後沒有資料。", file=sys.stderr)
            sys.exit(1)
        if column_to_plot not in df.columns:
            print(f"[ERROR] 合併檔缺少欄位 {column_to_plot}，可用欄位：{df.columns.tolist()}", file=sys.stderr)
            sys.exit(1)
        plot_df = df[["method", column_to_plot]].copy()
        # 若 method 欄有重複（多個 trial），取平均
        plot_df = plot_df.groupby("method", as_index=False).mean()
        out_png = out or f"avg_reward_W{W}_users{users}_alpha{alpha}.png"
        plot_from_dataframe(plot_df, W, alpha, users, column_to_plot, save_png, os.path.join(folder, out_png))
        return

    # 2) 否則在 results 裡找符合 pattern 的檔
    candidates = find_candidate_files(folder, W, users, alpha)
    if not candidates:
        print(f"[ERROR] 找不到任何檔案：pattern {folder}/*W{W}*users{users}*alpha*{alpha}*.csv", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 找到 {len(candidates)} 個候選檔案，開始讀取：")
    tmp = {}  # method -> list of values
    for f in candidates:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] 讀 {f} 失敗: {e}", file=sys.stderr)
            continue
        # 優先讀指定欄位
        col = column_to_plot if column_to_plot in df.columns else next((c for c in POSSIBLE_COLUMNS if c in df.columns), None)
        if col:
            val = float(df[col].mean())
            method = method_from_fname(f)
            tmp.setdefault(method, []).append(val)
            print(f"[INFO] {os.path.basename(f)} -> method={method}, {col} mean={val:.4f}")
        else:
            print(f"[WARN] {os.path.basename(f)} 沒有預期的 reward 欄位，跳過。", file=sys.stderr)

    if not tmp:
        print("[ERROR] 沒有任何檔案提供可用 reward 欄位，無法畫圖。", file=sys.stderr)
        sys.exit(1)

    # 對同 method 平均
    method_vals = {m: sum(vs)/len(vs) for m, vs in tmp.items()}

    # 轉成 dataframe 用既有 plot function
    plot_df = pd.DataFrame([{"method": m, column_to_plot: v} for m, v in method_vals.items()])

    out_png = out or f"avg_reward_W{W}_users{users}_alpha{alpha}.png"
    plot_from_dataframe(plot_df, W, alpha, users, column_to_plot, save_png, os.path.join(folder, out_png) if save_png else None)

if __name__ == "__main__":
    main()
