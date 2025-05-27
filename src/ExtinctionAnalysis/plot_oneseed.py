import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from hematopoiesis_model_v5 import simulate_hematopoiesis, params

initial_lt_hsc_count = 250
initial_st_hsc_count = 750
initial_lt_quiescent = 5
initial_st_quiescent = 5
time_steps = 5 * 40
sample_days = list(range(0, time_steps + 1, 5))  
num_seeds = 10

paa_values = [0.05]  # PAA values to test

plt.figure(figsize=(10, 6))

for paa in paa_values:
    params["PAA"] = paa
    #params["PAQA"] = paa/2

    print(f"Running simulations for PAA = {paa:.2f}")
    for seed in range(num_seeds):
        counts, _, _, _, _ = simulate_hematopoiesis(
            initial_lt_hsc_count,
            initial_st_hsc_count,
            params,
            time_steps,
            sample_days,
            initial_lt_quiescent,
            initial_st_quiescent,
            seed=seed,
        )
        # Calculate active LT-HSC at each sampled time point
        active_lt_hsc = np.array(counts["LT-HSC_active_labeled"]) + np.array(counts["LT-HSC_active_unlabeled"])
        print("len(sample_days):", len(sample_days))
        print("len(active_lt_hsc):", len(active_lt_hsc))
        n_points = len(active_lt_hsc)
        plt.plot(sample_days[:n_points], active_lt_hsc, label=f"PAA={paa:.2f}, seed={seed}")
        print(active_lt_hsc[-1])

plt.xlabel("Time (days)")
plt.ylabel("Active LT-HSC")
plt.title("Active LT-HSC vs Time")
plt.legend()
plt.grid(True)
plt.savefig("active_lt_hsc_vs_time.pdf", format="pdf", dpi=300)


plt.show()