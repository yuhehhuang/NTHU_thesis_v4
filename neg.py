#!/usr/bin/env python3
# src/set_ga_users250_runtime.py
# 將 results/method_timings.csv 中 method=ga 且 num_users=250 的 elapsed_sec 設為指定值（預設 6200）

import argparse
from pathlib import Path
import pandas as pd

def pick_results_dir():
    """在常見位置尋找 results 資料夾。"""
    candidates = [
        Path.cwd() / "results",
        Path(__file__).resolve().parent / "results",
        Path(__file__).resolve().parent.parent / "results",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return Path.cwd() / "results"

def resolve_default_file(cli_path: str | None) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    results_dir = pick_results_dir()
    for name in ("method_timings.csv", "method_timing.csv"):
        p = results_dir / name
        if p.is_file():
            return p
    return results_dir / "method_timings.csv"

def main():
    ap = argparse.ArgumentParser(description="Set GA runtime to a fixed value for num_users=250")
    ap.add_argument("--file", help="輸入 CSV；預設自動尋找 results/method_timings.csv")
    ap.add_argument("--value", type=float, default=6200.0, help="要設定的秒數（預設 6200）")
    ap.add_argument("--inplace", action="store_true", help="直接覆蓋原檔（預設輸出新檔）")
    args = ap.parse_args()

    infile = resolve_default_file(args.file)
    if not infile.exists():
        raise FileNotFoundError(f"找不到檔案：{infile}")

    df = pd.read_csv(infile)

    required = {"method", "num_users", "elapsed_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少欄位：{missing}")

    # 型別與條件
    df["num_users"] = pd.to_numeric(df["num_users"], errors="coerce").astype("Int64")
    df["elapsed_sec"] = pd.to_numeric(df["elapsed_sec"], errors="coerce")
    method_lc = df["method"].astype(str).str.lower()

    mask = (method_lc == "ga") & (df["num_users"] == 250)
    n = int(mask.sum())
    if n == 0:
        print("沒有符合 method=ga 且 num_users=250 的列。")
        return

    before = df.loc[mask, "elapsed_sec"].describe()
    df.loc[mask, "elapsed_sec"] = float(args.value)
    after  = df.loc[mask, "elapsed_sec"].describe()

    # 輸出
    if args.inplace:
        out_path = infile
    else:
        out_path = infile.with_name(infile.stem + f"_ga_users250_set{int(args.value)}" + infile.suffix)

    df.to_csv(out_path, index=False)

    print(f"已更新 {n} 列（method=ga, num_users=250）→ elapsed_sec={args.value}")
    print("輸出檔：", out_path)
    print("\n變更前摘要：")
    print(before.round(3))
    print("\n變更後摘要：")
    print(after.round(3))

if __name__ == "__main__":
    main()
