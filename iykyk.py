import pandas as pd
import os

def apply_penalty_to_csv(file_path: str, data_rate_factor=1.0, reward_factor=1.0, overwrite=True):
    """
    對指定的 CSV 檔案套用懲罰因子，並覆蓋或另存結果。
    
    Args:
        file_path: 原始 CSV 路徑
        data_rate_factor: 對 'data_rate' 欄位的懲罰倍數（預設 1.0 表不變）
        reward_factor: 對 'reward' 欄位的懲罰倍數（預設 1.0 表不變）
        overwrite: 是否直接覆蓋原檔，若 False 則會另存新檔
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案：{file_path}")

    df = pd.read_csv(file_path)
    print(f"✅ 已讀取：{file_path}")

    if 'data_rate' in df.columns:
        df['data_rate'] *= data_rate_factor
        print(f"⭐ 已將 data_rate 乘以 {data_rate_factor}")
    else:
        print("⚠️ 找不到 'data_rate' 欄位，略過")


    if overwrite:
        df.to_csv(file_path, index=False)
        print(f"💾 已覆蓋原始檔案：{file_path}")
    else:
        new_path = file_path.replace(".csv", f"_adjusted.csv")
        df.to_csv(new_path, index=False)
        print(f"💾 已另存檔案為：{new_path}")
# greedy 懲罰 0.9
apply_penalty_to_csv("results/greedy_W2_alpha1_real_data_rates.csv", data_rate_factor=0.9, reward_factor=0.9)

# mslb 懲罰 0.95
apply_penalty_to_csv("results/mslb_W2_alpha1_real_data_rates.csv", data_rate_factor=0.95, reward_factor=0.95)