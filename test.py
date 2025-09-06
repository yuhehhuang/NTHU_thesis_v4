import pandas as pd
import numpy as np

# -------------------------------
# 測試用參數
# -------------------------------
params = {
    "eirp_linear": 10 ** (37.7 / 10),        # 37.7 dBW
    "grx_linear": 10 ** (37.7 / 10),         # 37.7 dBi
    "boltzmann": 1.38e-23,
    "noise_temperature_k": 290,
    "channel_bandwidth_hz": 80e6,          # 80 MHz
}
params["noise_watt"] = (
    params["boltzmann"] * params["noise_temperature_k"] * params["channel_bandwidth_hz"]
)

# -------------------------------
# 計算 SINR 與 Data Rate
# -------------------------------
def compute_sinr_and_rate(params, PL_dB, PL_other_dB=None, interference_flag=True):
    PL_linear = 10 ** (PL_dB / 10)
    P_rx = params["eirp_linear"] * params["grx_linear"] / PL_linear

    interference = 0
    if interference_flag and PL_other_dB is not None:
        PL_other_linear = 10 ** (PL_other_dB / 10)
        interference = params["eirp_linear"] * params["grx_linear"] / PL_other_linear

    SINR = P_rx / (params["noise_watt"] + interference)
    data_rate = params["channel_bandwidth_hz"] * np.log2(1 + SINR) / 1e6  # Mbps
    return SINR, data_rate

# -------------------------------
# 測試不同條件
# -------------------------------
cases = []
for PL_dB in [180, 185, 190]:
    # 無干擾
    sinr_no_int, rate_no_int = compute_sinr_and_rate(params, PL_dB, interference_flag=False)
    cases.append(["No Interference", PL_dB, None, sinr_no_int, rate_no_int])

    # 有干擾 (相同 path loss)
    sinr_int, rate_int = compute_sinr_and_rate(params, PL_dB, PL_dB, interference_flag=True)
    cases.append(["With Interference", PL_dB, PL_dB, sinr_int, rate_int])

    # 有干擾 (干擾衛星更近，PL=180)
    sinr_int2, rate_int2 = compute_sinr_and_rate(params, PL_dB, 180, interference_flag=True)
    cases.append(["With Interference (Stronger)", PL_dB, 180, sinr_int2, rate_int2])

df_results = pd.DataFrame(
    cases,
    columns=["Case", "PL_dB (Signal)", "PL_dB (Interference)", "SINR", "Data Rate (Mbps)"]
)

# -------------------------------
# 輸出結果
# -------------------------------
import math
print(f"\nNoise Power (W): {params['noise_watt']:.3e}")
print(f"Noise Power (dBW): {10 * math.log10(params['noise_watt']):.2f} dBW")
print("Path Losses Tested: 180 dB, 185 dB, 190 dB")
print("\n===== SINR & Data Rate 測試結果 =====\n")
print(df_results.to_string(index=False))