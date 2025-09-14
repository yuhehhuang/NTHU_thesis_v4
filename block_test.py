#!/usr/bin/env python3  # 指定此檔可由系統的 python 執行器執行
# block_test.py  # 檔案名稱（說明用途：模擬 blocking rate）

# --- 匯入標準函式庫與第三方套件 ---
import os  # 作業系統路徑與檔案操作
import glob  # 檔案 pattern 搜尋
import argparse  # 命令列參數解析
import random  # 隨機函數
import time  # 時間相關（未大量使用，但保留）
from datetime import datetime  # 取得格式化時間字串
import pandas as pd  # 資料處理 (DataFrame, CSV)
import numpy as np  # 數值/亂數種子一致性

# --- 匯入專案內的 load_system_data 函式（需在 src/init.py） ---
from src.init import load_system_data  # 取得系統資料（users, access_matrix, params, backup 等）

# --------------------------
# 常數與小工具：prefix 解析與路徑找檔
# --------------------------
KNOWN_SUFFIXES = [  # 已知檔案副檔名/後綴清單，用於從檔名剝離出 base prefix
    "_load_by_time.csv",  # load_by_time CSV 後綴
    "_paths.csv",  # paths CSV 後綴
    "_results.csv",  # results CSV 後綴
    "_data_rates.csv",  # data_rates CSV 後綴
    "_real_data_rates.csv",  # real data rates CSV 後綴
    "_blocking_summary.csv",  # blocking summary 後綴
    "_blocking_details.csv",  # blocking details 後綴
    ".csv"  # 通用 csv 後綴
]

def resolve_prefix_prefer_load(prefix, results_dir="results"):
    """
    優先尋找 results/{prefix}_load_by_time.csv 的函式。
    若存在就回傳 prefix；否則嘗試其他 exact suffix；
    若仍找不到，才做 wildcard 搜尋（與原來行為相同）。
    """
    # 1) 如果存在精確的 load_by_time 檔案，直接回傳原始 prefix（最優先）
    pref_file = os.path.join(results_dir, prefix + "_load_by_time.csv")
    if os.path.exists(pref_file):
        return prefix

    # 2) 再嘗試其他 known suffix 的 exact match（如果有任何一個存在就回傳 prefix）
    for s in KNOWN_SUFFIXES:
        p = os.path.join(results_dir, prefix + s)
        if os.path.exists(p):
            return prefix

    # 3) 如果 exact match 都沒找到，建立一些變體（_user <-> _users）然後做 wildcard 搜尋
    variants = [prefix]
    if "_user" in prefix and "_users" not in prefix:
        variants.append(prefix.replace("_user", "_users"))
    if "_users" in prefix and "_user" not in prefix:
        variants.append(prefix.replace("_users", "_user"))

    # 蒐集所有候選檔案
    candidates = []
    for v in variants:
        candidates += glob.glob(os.path.join(results_dir, v + "*"))

    # 4) 如果沒有任何候選，回傳原始 prefix（fallback，不改變行為）
    if not candidates:
        return prefix

    # 5) 選取最新修改時間的 candidate，並把已知檔尾移除以取得 base prefix
    newest = max(candidates, key=os.path.getmtime)
    base = os.path.basename(newest)
    for s in KNOWN_SUFFIXES:
        if base.endswith(s):
            base = base[:-len(s)]
            break
    return base
resolve_prefix = resolve_prefix_prefer_load

# --------------------------
# 讀取 results 裡的 load_by_time（優先使用此檔）
# --------------------------
def load_load_by_time_from_results(prefix):  # 從 results/{prefix}_load_by_time.csv 載入負載時間序列
    path = os.path.join("results", f"{prefix}_load_by_time.csv")  # 建立檔案路徑
    if not os.path.exists(path):  # 若檔案不存在
        return None  # 回傳 None 表示找不到
    df = pd.read_csv(path)  # 讀取 CSV 為 DataFrame
    T = int(df["time"].max()) + 1  # 推斷 time slot 數量（最大 time + 1）
    load_by_time = [dict() for _ in range(T)]  # 建立長度為 T 的字典清單（每項代表一個時槽）
    for _, row in df.iterrows():  # 逐列處理
        t = int(row["time"])  # 取得 time
        sat = row["sat"]  # 取得衛星名稱
        load = int(row["load"])  # 取得負載數
        load_by_time[t][sat] = load  # 寫入對應時槽的衛星負載
    return load_by_time  # 回傳重建的 load_by_time

# 若 load_by_time 不存在，嘗試從 paths.csv 重建
def reconstruct_load_by_time_from_paths(prefix, T_guess=None):  # 從 results/{prefix}_paths.csv 重新計數以構建 load_by_time
    path = os.path.join("results", f"{prefix}_paths.csv")  # 建立 paths 檔案路徑
    if not os.path.exists(path):  # 若沒找到 paths.csv
        return None  # 回傳 None
    df = pd.read_csv(path)  # 讀取 paths.csv
    if "time" not in df.columns or "sat" not in df.columns:  # 檢查必要欄位是否存在
        return None  # 欄位不足則無法重建
    T = int(df["time"].max()) + 1 if T_guess is None else T_guess  # 若沒給 T_guess，從 paths.csv 推斷 T
    load_by_time = [dict() for _ in range(T)]  # 建立空的 load_by_time 結構
    for _, r in df.iterrows():  # 逐列走訪 paths.csv
        t = int(r["time"])  # 取得 time
        sat = r["sat"]  # 取得衛星名稱
        load_by_time[t][sat] = load_by_time[t].get(sat, 0) + 1  # 對 (t,sat) 計數
    return load_by_time  # 回傳重建結果

# --------------------------
# 產生背景使用者清單：支援三種模式
# --------------------------
def gen_bg_users(n_bg, T, min_dur=5, max_dur=12, sats=None, df_access=None, mode="stick_until_invisible"):
    """
    產生背景 users：
      mode:
        - fixed_sat: 整段堅持 initial sat（不檢查可見性）
        - any_visible: 每個 slot 隨機從該 slot 可見衛星選一顆
        - stick_until_invisible: 選一顆直到不可見才換手
    回傳 list of dict，每個 dict 含 id,t_start,t_end,sat,per_slot_sats
    """
    import ast  # 解析若 visible_sats 存為字串的情況
    users = []  # 儲存背景使用者的 list

    def visible_sats_at(t):  # 取得第 t 時槽的可見衛星清單
        if df_access is None:  # 若沒有提供 access matrix
            return list(sats) if sats is not None else []  # 回傳參數 sats 或空 list
        try:
            row = df_access.loc[t]  # 取第 t 列
            vs = row["visible_sats"]  # 取得 visible_sats 欄位
            if isinstance(vs, str):  # 若為字串表示
                try:
                    vs = ast.literal_eval(vs)  # 嘗試把字串轉為 list
                except Exception:
                    vs = [s.strip() for s in vs.split(",") if s.strip()]  # 退化處理：逗號分隔
            return list(vs)  # 回傳 list
        except Exception:
            return list(sats) if sats is not None else []  # 發生例外時退回 sats 或空 list

    for i in range(n_bg):  # 產生 n_bg 個背景使用者
        t_start = random.randint(0, max(0, T - 1))  # 隨機起始時槽（0..T-1）
        dur = random.randint(min_dur, max(1, min(max_dur, T - t_start)))  # 隨機持續時間（受 T 限制）
        t_end = min(T, t_start + dur)  # 結束時槽（exclusive）

        vis0 = visible_sats_at(t_start)  # 取得 t_start 時的可見衛星
        if vis0:
            initial_sat = random.choice(vis0)  # 若有可見衛星則從中選一顆
        else:
            initial_sat = random.choice(sats) if sats else None  # 否則從所有 sats 選或 None

        per_slot_sats = []  # 為每個 slot 建立使用衛星清單
        cur_sat = initial_sat  # 當前使用的衛星（在 stick_until_invisible 模式下維持）

        for t in range(t_start, t_end):  # 為每個時槽決定衛星
            vis = visible_sats_at(t)  # 該時槽可見衛星

            if mode == "any_visible":  # 每 slot 隨機選可見衛星
                if vis:
                    per_slot_sats.append(random.choice(vis))  # 選一顆可見衛星
                else:
                    per_slot_sats.append(None)  # 無可見衛星則 None
                continue  # 處理下一時槽

            if mode == "fixed_sat":  # 整段使用 initial_sat（不檢查可見性）
                per_slot_sats.append(initial_sat)  # 直接加入初始衛星
                continue  # 下一時槽

            # 以下為 stick_until_invisible 邏輯
            if cur_sat is not None and cur_sat in vis:  # 若目前衛星在該 slot 仍可見
                per_slot_sats.append(cur_sat)  # 繼續使用
                continue  # 下一時槽

            if vis:  # 若目前衛星不可見且此 slot 有可見衛星
                new_sat = random.choice(vis)  # 換手：隨機選一顆可見衛星
                per_slot_sats.append(new_sat)  # 記錄該 slot 使用的新衛星
                cur_sat = new_sat  # 更新當前衛星
            else:
                per_slot_sats.append(None)  # 無可見衛星，該 slot 無服務
                cur_sat = None  # 無當前衛星

        users.append({  # 把此使用者封裝成 dict 並加入列表
            "id": f"BG{i}",  # 背景使用者 id
            "t_start": t_start,  # 起始時槽
            "t_end": t_end,  # 結束時槽（exclusive）
            "sat": initial_sat,  # 初始衛星（相容舊版）
            "per_slot_sats": per_slot_sats  # 每時槽實際使用的衛星或 None
        })
    return users  # 回傳背景使用者清單

# --------------------------
# 單次 blocking 模擬主程式
# --------------------------
def run_blocking_trial(load_by_time_base, sat_channel_dict_backup, params, df_access,
                       bg_users, mode="fixed_sat", num_channels=25):
    """
    將 bg_users 按序嘗試加入現有的 load_by_time（working），若任何 slot 無可用 channel 則該 user 被 blocked。
    回傳 blocked 總數、total、blocking_rate 與詳細列表
    """
    T = len(load_by_time_base)  # 取得時間長度 T
    working = [dict(tdict) for tdict in load_by_time_base]  # deep copy 基礎負載（避免修改原始）

    # 計算每顆衛星的靜態 pre-occupied channel 數（來自 sat_channel_dict_backup）
    random_static_load = {}  # 儲存衛星: 靜態占用數
    for sat, chdict in (sat_channel_dict_backup or {}).items():  # 走訪 backup 結構
        try:
            random_static_load[sat] = int(sum(chdict.values()))  # 若 channel dict 是 0/1 值，sum 給出占用數
        except Exception:
            random_static_load[sat] = 0  # 若格式不符則設為 0

    blocked = 0  # 被阻擋用戶計數
    total = 0  # 總請求計數
    details = []  # 儲存每位 bg user 的結果明細

    # 建立系統中所有出現過的衛星清單（union）
    all_sats = set()  # 使用集合避免重複
    for tdict in working:  # 走訪每個時槽的已分配字典
        all_sats.update(tdict.keys())  # 把該時槽出現的衛星加入集合
    all_sats.update(random_static_load.keys())  # 加上 backup 中的衛星
    all_sats = sorted(list(all_sats))  # 轉為排序後的 list（方便查看與選取）

    def visible_sats_at(t):  # helper：從 df_access 取得第 t 時槽的可見衛星
        if df_access is None:
            return all_sats  # 若沒有 access matrix，回傳全部衛星做備援
        try:
            row = df_access.loc[t]  # 讀取第 t 列
            vs = row["visible_sats"]  # 取得 visible_sats 欄位
            if isinstance(vs, str):  # 若以字串儲存（CSV 常見）
                import ast  # 解析字串為 list
                try:
                    vs = ast.literal_eval(vs)  # 嘗試解析
                except Exception:
                    vs = [s.strip() for s in vs.split(",") if s.strip()]  # 退化處理
            return list(vs)  # 回傳 list
        except Exception:
            return all_sats  # 發生例外時回傳全部衛星做備援

    for u in bg_users:  # 對每個背景使用者依序嘗試分配
        total += 1  # 總請求數增加
        t0, t1 = u["t_start"], u["t_end"]  # 取得該 user 的時段（t0..t1-1）
        user_blocked = False  # 該 user 是否被阻擋的旗標
        allocated_slots = []  # 記錄已分配的 (t,sat) 以便失敗時回滾

        # 若 fixed_sat 模式但使用者沒有預指定 sat，則在 t0 從可見衛星中挑一顆
        if mode == "fixed_sat" and u["sat"] is None:
            vis = visible_sats_at(t0)  # 取得 t0 可見衛星
            if not vis:
                vis = all_sats  # 若沒可見衛星則 fallback 全衛星
            chosen_sat = random.choice(vis)  # 隨機選一顆
        else:
            chosen_sat = u.get("sat", None)  # 否則使用 u["sat"]（可能為 None）

        for t in range(t0, t1):  # 走訪該 user 的所有時槽
            # 優先使用 gen_bg_users 產生的 per_slot_sats（若存在）
            per_slot_sat = None  # 預設無 per-slot 指定
            if "per_slot_sats" in u:  # 若使用者包含 per_slot_sats 欄位
                idx = t - t0  # 計算在 per_slot_sats 的索引
                if 0 <= idx < len(u["per_slot_sats"]):  # 若索引合法
                    per_slot_sat = u["per_slot_sats"][idx]  # 取得該 slot 指定的衛星或 None

            if per_slot_sat is not None:  # 若該 slot 有指定衛星
                sat = per_slot_sat  # 以此衛星為目標
                assigned = working[t].get(sat, 0)  # 該 slot 已分配數
                static = random_static_load.get(sat, 0)  # backup 的靜態 load
                if assigned + static >= num_channels:  # 若 channel 已滿
                    user_blocked = True  # 標記被阻擋
                    break  # 終止該 user 分配
                working[t][sat] = assigned + 1  # 分配一個 channel
                allocated_slots.append((t, sat))  # 記錄已分配以便回滾
                continue  # 成功分配此 slot，處理下一 slot

            # 若沒有 per-slot 指定，回退到 mode 對應的分配邏輯
            if mode == "fixed_sat":  # 固定衛星模式
                sat = chosen_sat  # 指定衛星
                assigned = working[t].get(sat, 0)  # 取得已分配數
                static = random_static_load.get(sat, 0)  # 靜態佔用數
                if assigned + static >= num_channels:  # 若已滿
                    user_blocked = True  # 標記阻擋
                    break  # 終止
                working[t][sat] = assigned + 1  # 分配
                allocated_slots.append((t, sat))  # 記錄
            else:  # any_visible 或 stick_until_invisible（此處 stick 已在 per_slot_sats 處理）
                vis = visible_sats_at(t)  # 取得此 slot 的可見衛星
                if not vis:  # 若無可見衛星
                    user_blocked = True  # 被阻擋（沒有 fallback）
                    break
                sat_found = None  # 初始化找到的衛星
                for sat in random.sample(vis, len(vis)):  # 隨機順序檢查可見衛星
                    assigned = working[t].get(sat, 0)  # 該 slot 的已分配數
                    static = random_static_load.get(sat, 0)  # 靜態佔用
                    if assigned + static < num_channels:  # 若還有空位
                        sat_found = sat  # 記錄找到的衛星
                        break  # 跳出
                if sat_found is None:  # 如果沒找到任何有空位的衛星
                    user_blocked = True  # 被阻擋
                    break  # 終止分配
                working[t][sat_found] = working[t].get(sat_found, 0) + 1  # 分配到該衛星
                allocated_slots.append((t, sat_found))  # 記錄

        if user_blocked:  # 若使用者整段被阻擋
            blocked += 1  # 阻擋計數加一
            # 回滾已分配的 slots（假設該請求全數失敗）
            for (t, s) in allocated_slots:  # 逐一回滾
                working[t][s] = working[t].get(s, 1) - 1  # 減少之前加過的計數
                if working[t][s] <= 0:  # 若減到 0 或更小
                    working[t].pop(s, None)  # 移除該 key 以保持字典乾淨
            details.append({"id": u["id"], "blocked": True, "t_start": t0, "t_end": t1})  # 記錄細節
        else:
            details.append({"id": u["id"], "blocked": False, "t_start": t0, "t_end": t1})  # 記錄成功細節

    blocking_rate = blocked / total if total > 0 else 0.0  # 計算 blocking rate（避免除以 0）
    return blocked, total, blocking_rate, details  # 回傳結果

# --------------------------
# main: 執行多次 trial 並儲存結果
# --------------------------
def main(args):  # 主程式入口，接收解析好的 args
    random.seed(args.seed)  # 設定 random 模組種子以便重現
    np.random.seed(args.seed)  # 設定 numpy 種子

    # 簡易 debug 資訊輸出
    print(">>> BLOCK TEST START")  # 標記開始
    print(f"Requested prefix: {args.prefix}")  # 顯示使用者指定的 prefix
    if args.user_csv:  # 若使用者指定 user csv，顯示出來
        print(f"Requested user csv: {args.user_csv}")

    # 載入系統資料（若 load_system_data 支援 user_csv_path，這裡會傳入）
    try:
        system = load_system_data(regenerate_sat_channels=False, user_csv_path=args.user_csv)
    except TypeError:
        # 若你尚未修改 src/init.py，使其接受 user_csv_path，使用舊版呼叫（fallback）
        print("[WARN] load_system_data() does not accept user_csv_path. Calling without it.")
        system = load_system_data(regenerate_sat_channels=False)

    df_access = system.get("access_matrix", None)  # 取得 access matrix（可能為 None）
    sat_channel_dict_backup = system.get("sat_channel_dict_backup", {}) or {}  # 取得備援的 sat channel dict
    params = system.get("params", {}) or {}  # 取得系統參數（預設空 dict）
    num_channels = int(params.get("num_channels", params.get("num_channels_per_sat", 25)))  # 每顆衛星的 channel 數上限

    # 解析 prefix（可以模糊輸入短 prefix），實際使用的 prefix 存到 actual_prefix
    actual_prefix = resolve_prefix(args.prefix)
    if actual_prefix != args.prefix:
        print(f"[INFO] Resolved prefix '{args.prefix}' -> '{actual_prefix}' (using latest matching file in results/)")
    else:
        print(f"[INFO] Using prefix '{args.prefix}' (no matching timestamped file found)")

    # 優先從 results 讀取 load_by_time（若找不到則嘗試從 paths 重建）
    load_by_time = load_load_by_time_from_results(actual_prefix)
    T = None
    if load_by_time is not None:
        T = len(load_by_time)  # 若成功讀到，設定 T
    else:
        # 嘗試從 paths.csv 重建，利用 df_access 的長度作為猜測值
        T_guess = None
        if df_access is not None:
            try:
                T_guess = len(df_access)
            except Exception:
                T_guess = None
        load_by_time = reconstruct_load_by_time_from_paths(actual_prefix, T_guess)
        if load_by_time is not None:
            T = len(load_by_time)

    # fallback：若 load_by_time 仍為 None，則建立空的 load_by_time（所有 assigned load 預設 0）
    if load_by_time is None:
        if df_access is not None:
            T = len(df_access)  # 以 access matrix 的長度作為 T
        else:
            T = 100  # 否則預設 100
        load_by_time = [dict() for _ in range(T)]  # 建立空的時間序列

    # 準備 df_access_local，確保可以直接用 loc[t] 取得
    df_access_local = None
    if df_access is not None:
        try:
            df_access_local = df_access.reset_index(drop=True)  # 重設索引以確保整數索引對齊
        except Exception:
            df_access_local = df_access  # 若失敗則直接使用原本的 df_access

    # 整合所有衛星名稱（來自 backup 與 load_by_time）
    sats = sorted(list(set(list(sat_channel_dict_backup.keys()) + [
        s for tdict in load_by_time for s in tdict.keys()
    ])))  # 產生排序後的衛星清單

    # 執行多個 trial，收集 summary 與 details
    results = []  # summary list
    details_all = []  # 詳細紀錄 list
    for trial in range(args.trials):  # 迴圈執行 args.trials 次獨立模擬
        n_bg = args.n_bg  # 背景使用者數量
        bg_users = gen_bg_users(n_bg=n_bg, T=T, min_dur=args.min_dur, max_dur=args.max_dur,
                                sats=sats, df_access=df_access_local, mode=args.mode)  # 產生背景使用者
        blocked, total, blocking_rate, details = run_blocking_trial(
            load_by_time_base=load_by_time,  # 傳入基礎負載
            sat_channel_dict_backup=sat_channel_dict_backup,  # 傳入 backup
            params=params,  # 傳入系統參數
            df_access=df_access_local,  # 傳入 access matrix
            bg_users=bg_users,  # 背景使用者清單
            mode=args.mode,  # 分配模式
            num_channels=num_channels  # 每衛星頻道上限
        )
        tstamp = datetime.now().isoformat(timespec="seconds")  # 取得時間戳記
        results.append({  # 收集本 trial 的 summary
            "timestamp": tstamp,
            "prefix": actual_prefix,
            "trial": trial,
            "n_bg": n_bg,
            "min_dur": args.min_dur,
            "max_dur": args.max_dur,
            "mode": args.mode,
            "blocked": blocked,
            "total": total,
            "blocking_rate": blocking_rate
        })
        for d in details:  # 將本 trial 的每位使用者細節加入 details_all（並附 metadata）
            drec = dict(d)  # 複製 detail dict
            drec.update({"prefix": actual_prefix, "trial": trial, "mode": args.mode, "timestamp": tstamp})
            details_all.append(drec)  # 加入到總細節清單

    # 儲存 summary 與 details 到 results/
    os.makedirs("results", exist_ok=True)  # 確保 results 資料夾存在
    out_sum = os.path.join("results", f"{actual_prefix}_blocking_summary.csv")  # summary 輸出路徑
    out_det = os.path.join("results", f"{actual_prefix}_blocking_details.csv")  # details 輸出路徑
    pd.DataFrame(results).to_csv(out_sum, index=False)  # 寫 summary CSV
    pd.DataFrame(details_all).to_csv(out_det, index=False)  # 寫 details CSV

    # 印出簡短的聚合統計並告知檔案位置
    df_res = pd.DataFrame(results)  # 轉為 DataFrame 以便聚合
    try:
        print(df_res.groupby("mode")["blocking_rate"].agg(["mean", "std", "count"]))  # 印出每種 mode 的平均/標準差/次數
    except Exception:
        print("No results to aggregate.")  # 若結果為空則提示
    print(f"Saved summary: {out_sum}")  # 顯示 summary 存放路徑
    print(f"Saved details: {out_det}")  # 顯示 details 存放路徑

# --------------------------
# CLI 參數解析器（腳本執行時使用）
# --------------------------
if __name__ == "__main__":  # 若直接以腳本執行（非匯入）
    parser = argparse.ArgumentParser()  # 建立參數解析器
    parser.add_argument("--prefix", required=True,
                        help="對應 main.py 儲存的 prefix，例如 'dp_W3_users125_alpha1' 或簡短 'dp_W3_user125_alpha1'（會自動模糊比對 results/）")  # prefix 參數說明
    parser.add_argument("--user_csv", type=str, default=None,
                        help="要讀取的 user csv，例如 data/user_info125.csv（可選，若不指定 load_system_data 會用預設或自動選最新 user 檔）")  # 可選 user csv 路徑
    parser.add_argument("--n_bg", type=int, default=100, help="每 trial background users 數量")  # 背景 user 數量
    parser.add_argument("--min_dur", type=int, default=1, help="背景使用者最短持續時間")  # 最短持續時槽
    parser.add_argument("--max_dur", type=int, default=10, help="背景使用者最長持續時間")  # 最長持續時槽
    parser.add_argument("--trials", type=int, default=20, help="要跑幾個 trial（獨立模擬）")  # trial 次數
    parser.add_argument("--mode", choices=["fixed_sat", "any_visible", "stick_until_invisible"],
                        default="stick_until_invisible",
                        help=("fixed_sat: 整段使用同一顆初始衛星，不檢查可見性；"
                              "any_visible: 每 slot 隨機選當下可見衛星；"
                              "stick_until_invisible: t_start 選一顆並持續使用，僅當該衛星在某 slot 不可見時才換手"))  # 支援三種分配模式
    parser.add_argument("--seed", type=int, default=42, help="亂數種子（預設 42）")  # 種子參數
    args = parser.parse_args()  # 解析命令列參數
    main(args)  # 呼叫主程序並傳入解析後的參數
