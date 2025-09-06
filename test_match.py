import pandas as pd

# === 載入你的 hungarian_W2_alpha1_real_data_rates.csv ===
df = pd.read_csv("results/hungarian_W2_alpha1_real_data_rates.csv")

# ✅ 基本檢查：哪幾行 data rate 為 0
df_zero = df[df["data_rate"] == 0]

# ✅ 統計每個 (time, channel) 上的使用人數
usage_count = df.groupby(["time", "channel"]).size().reset_index(name="user_count")

# ✅ 統計 data_rate = 0 的集中程度
zero_count = df_zero.groupby(["time", "channel"]).size().reset_index(name="zero_count")

# ✅ 合併資訊
merged = pd.merge(usage_count, zero_count, how="left", on=["time", "channel"])
merged["zero_count"] = merged["zero_count"].fillna(0).astype(int)
merged["zero_ratio"] = merged["zero_count"] / merged["user_count"]

# ✅ 篩出可能干擾嚴重的情況（例如 50% 以上 user 都是 0）
problem_cases = merged[merged["zero_ratio"] > 0.5]

# === 顯示結果 ===
print("🔍 每個時間與 channel 的使用者數量與 0-rate 使用者比例：")
print(merged.sort_values(by=["zero_ratio"], ascending=False).head(20))

print("\n⚠️ 疑似嚴重干擾的情況（zero_ratio > 0.5）：")
print(problem_cases)

# ✅ 若你想另存成檔案：
merged.to_csv("channel_usage_analysis.csv", index=False)
