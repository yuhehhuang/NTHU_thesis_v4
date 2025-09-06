# user.py
import pandas as pd
import random
from geopy.distance import distance
from geopy.point import Point
import argparse
import os

def make_users(num_users, num_groups=25, center_lat=40.0386, center_lon=-75.5966, T_slots=10*60//15):
    NUM_USERS = num_users
    NUM_GROUPS = num_groups
    USERS_PER_GROUP = max(1, NUM_USERS // NUM_GROUPS)

    center = Point(center_lat, center_lon)
    users = []

    # 產生所有可能的開始時間（保留至少 12 slots 空間）
    possible_starts = list(range(0, max(0, T_slots - 12) + 1))
    # 若 groups > possible starts，允許重複選取
    if NUM_GROUPS <= len(possible_starts):
        group_starts = random.sample(possible_starts, NUM_GROUPS)
    else:
        group_starts = [random.choice(possible_starts) for _ in range(NUM_GROUPS)]

    uid = 0
    for group_id in range(NUM_GROUPS):
        group_t_start = group_starts[group_id]
        for i in range(USERS_PER_GROUP):
            if uid >= NUM_USERS:
                break
            duration = random.randint(10, 12)  # 10~12 slots
            t_end = min(group_t_start + duration, T_slots - 1)
            angle = random.uniform(0, 360)
            radius = random.uniform(0, 0.8)  # km
            point = distance(kilometers=radius).destination(center, bearing=angle)
            users.append({
                "user_id": int(uid),
                "t_start": int(group_t_start),
                "t_end": int(t_end),
                "lat": float(point.latitude),
                "lon": float(point.longitude)
            })
            uid += 1

    # 如果分組產生不足 num_users（因為整數除法），再補 random users
    while len(users) < NUM_USERS:
        group_t_start = random.choice(possible_starts)
        duration = random.randint(10, 12)
        t_end = min(group_t_start + duration, T_slots - 1)
        angle = random.uniform(0, 360)
        radius = random.uniform(0, 0.8)
        point = distance(kilometers=radius).destination(center, bearing=angle)
        users.append({
            "user_id": int(uid),
            "t_start": int(group_t_start),
            "t_end": int(t_end),
            "lat": float(point.latitude),
            "lon": float(point.longitude)
        })
        uid += 1

    df = pd.DataFrame(users)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_users", type=int, default=100, help="要產生的 user 數")
    p.add_argument("--out", type=str, default=None, help="輸出檔名（會放在 data/ 下），如 user_info100.csv；若未指定，會自動使用 data/user_info{N}.csv")
    args = p.parse_args()

    os.makedirs("data", exist_ok=True)
    df = make_users(num_users=args.num_users)

    out_fn = args.out if args.out else f"data/user_info{args.num_users}.csv"
    df.to_csv(out_fn, index=False)
    print(f"✅ 產生 {out_fn} （共 {len(df)} users）")

if __name__ == "__main__":
    main()
