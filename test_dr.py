#!/usr/bin/env python3
# test.py
# 用法：
#   python test_dr.py results/dp_W3_users100_alpha0.25_real_data_rates.csv
#   （也可一次丟多個檔案）

import os
import sys
import argparse
import pandas as pd
import numpy as np

def compute_average_data_rate(csv_path: str) -> float:
    """讀取 csv 並回傳整體平均 data_rate（忽略 NaN）"""
    df = pd.read_csv(csv_path)
    if "data_rate" not in df.columns:
        raise ValueError(f"{csv_path}: 缺少 'data_rate' 欄位")
    rates = pd.to_numeric(df["data_rate"], errors="coerce")
    rates = rates.dropna()
    if rates.empty:
        return float("nan")
    return float(rates.mean())

def main():
    ap = argparse.ArgumentParser(description="計算 real_data_rates 檔案的平均 data_rate")
    ap.add_argument("csv", nargs="+", help="輸入 CSV 路徑（可多個）")
    args = ap.parse_args()

    exit_code = 0
    for path in args.csv:
        try:
            avg = compute_average_data_rate(path)
            print(f"{os.path.basename(path)} average_data_rate = {avg:.6f}")
        except Exception as e:
            print(f"[ERROR] {path}: {e}", file=sys.stderr)
            exit_code = 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
