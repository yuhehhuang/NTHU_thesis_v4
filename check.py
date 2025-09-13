import pandas as pd

df = pd.read_csv("results/hungarian_W3_users250_alpha0_paths.csv")  # 換成你的檔案路徑
present = set(df["user_id"])
missing = sorted(set(range(250)) - present)
print("總筆數:", len(df), "獨特 user 數:", df["user_id"].nunique())
print("缺少的 user_id:", missing)          # 會印出 [] 表示沒有缺