import pandas as pd
import matplotlib.pyplot as plt
import ast

df = pd.read_csv("data/access_matrix.csv")
df["visible_sats"] = df["visible_sats"].apply(ast.literal_eval)  # 轉成真正的 list

num_visible_sats = df["visible_sats"].apply(len)

plt.figure(figsize=(8, 5))
plt.bar(df["time_slot"], num_visible_sats)  # 用 time_slot 而不是 df.index
plt.xlabel("Time Slot")
plt.ylabel("Number of Accessible Satellites")
plt.title("Number of Accessible Satellites per Time Slot")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/num_accessible_sats_per_timeslot.png")
plt.show()
