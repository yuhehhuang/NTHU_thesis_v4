#!/usr/bin/env python3
"""
revise.py

用途：
  - 掃描 results/ 下的已存在實驗檔案，為每組檔案建立新的複本，
    在 prefix 中插入 "_users{N}"，例如：
      dp_W3_alpha1_load_by_time.csv  -> dp_W3_users100_alpha1_load_by_time.csv
      hungarian_W3_alpha1_20250904T153012_paths.csv ->
        hungarian_W3_users100_alpha1_20250904T153012_paths.csv

說明：
  - 預設只做「預覽列出」改名結果（--apply 才會真的建立複本）
  - 預設不覆蓋已存在的目標檔（除非使用 --overwrite）
  - 支援多種常見 suffix（load_by_time, paths, results, data_rates, real_data_rates, blocking_summary, blocking_details）
使用範例：
  # 只預覽
  python revise.py --user_count 100

  # 實際建立複本（results/ 下會產生新檔）
  python revise.py --user_count 100 --apply

  # 強制覆蓋已存在目標檔
  python revise.py --user_count 100 --apply --overwrite

  # 只處理特定 method/W 組（如只處理dp_W3開頭檔案）
  python revise.py --user_count 100 --apply --filter_prefix dp_W3
"""
import os
import re
import shutil
import argparse

# 可被處理的檔案後綴（會嘗試匹配 endswith）
KNOWN_SUFFIXES = [
    "_load_by_time.csv",
    "_paths.csv",
    "_results.csv",
    "_data_rates.csv",
    "_real_data_rates.csv",
    "_blocking_summary.csv",
    "_blocking_details.csv"
]

def find_suffix(filename):
    """回傳匹配到的已知 suffix，或 None"""
    for s in KNOWN_SUFFIXES:
        if filename.endswith(s):
            return s
    return None

def insert_users_segment(base_prefix, user_count):
    """
    在 base_prefix（不包含 suffix 的部份）中插入 _users{user_count}。
    插入位置優先：在第一個 '_W<number>' 之後插入；若找不到則在 timestamp 或末尾插入。
    """
    # 若已包含 usersXXX，直接回傳原本（避免重複插入）
    if re.search(r"_users\d+", base_prefix):
        return base_prefix

    # 找到 _W\d+ 的位置
    m = re.search(r"_W\d+", base_prefix)
    if m:
        insert_pos = m.end()
        newbase = base_prefix[:insert_pos] + f"_users{user_count}" + base_prefix[insert_pos:]
        return newbase

    # 若沒找到 _W\d+，嘗試在 timestamp 前插入（timestamp 格式例：_20250904T153012）
    m2 = re.search(r"_\d{8}T\d{6}", base_prefix)
    if m2:
        insert_pos = m2.start()
        newbase = base_prefix[:insert_pos] + f"_users{user_count}" + base_prefix[insert_pos:]
        return newbase

    # fallback：直接在末尾插入
    return base_prefix + f"_users{user_count}"

def build_renames(results_dir, user_count, filter_prefix=None):
    """
    掃描 results_dir，建立 (src_path, dst_path) 的清單。
    filter_prefix: 若提供，僅處理檔名以該字串開頭的檔案（不包含 suffix）
    """
    renames = []
    for fn in os.listdir(results_dir):
        src = os.path.join(results_dir, fn)
        if not os.path.isfile(src):
            continue
        suffix = find_suffix(fn)
        if suffix is None:
            continue
        base = fn[:-len(suffix)]  # 去掉 suffix 的部份
        if filter_prefix is not None and not base.startswith(filter_prefix):
            # skip 不符合 filter 的
            continue
        newbase = insert_users_segment(base, user_count)
        dst_fn = newbase + suffix
        dst = os.path.join(results_dir, dst_fn)
        # 如果 src 跟 dst 相同（表示已經包含 usersXXX），跳過
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        renames.append((src, dst))
    return renames

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results", help="results 資料夾位置（預設 results）")
    p.add_argument("--user_count", type=int, required=True, help="要插入的 user 數，例如 100")
    p.add_argument("--apply", action="store_true", help="若指定則實際建立複本；否則只做預覽")
    p.add_argument("--overwrite", action="store_true", help="若指定且 --apply，則會覆蓋目標檔")
    p.add_argument("--filter_prefix", default=None, help="只處理以此 base_prefix 開頭的檔案（例如 'dp_W3'）")
    args = p.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"錯誤：找不到 results 資料夾（{results_dir}）")
        return

    renames = build_renames(results_dir, args.user_count, filter_prefix=args.filter_prefix)
    if not renames:
        print("沒有找到符合條件的檔案可以處理。")
        return

    print("以下為預計建立的複本清單（來源 → 目標）：")
    for src, dst in renames:
        print(f"  {os.path.basename(src)}  ->  {os.path.basename(dst)}")

    if not args.apply:
        print("\n目前為預覽模式（未實際建立檔案）。如要實際建立，請加上 --apply")
        return

    # 實際建立複本
    for src, dst in renames:
        if os.path.exists(dst):
            if args.overwrite:
                try:
                    shutil.copy2(src, dst)
                    print(f"覆蓋並複製：{os.path.basename(src)} -> {os.path.basename(dst)}")
                except Exception as e:
                    print(f"複製失敗：{src} -> {dst}；錯誤：{e}")
            else:
                print(f"跳過（目標已存在）：{os.path.basename(dst)}")
        else:
            try:
                shutil.copy2(src, dst)
                print(f"複製：{os.path.basename(src)} -> {os.path.basename(dst)}")
            except Exception as e:
                print(f"複製失敗：{src} -> {dst}；錯誤：{e}")

    print("\n完成。請檢查 results/ 中的新檔案，然後在你的 block_test.py 呼叫時使用新的 prefix（不帶 _load_by_time.csv）。")

if __name__ == "__main__":
    main()
