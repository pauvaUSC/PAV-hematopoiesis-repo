import numpy as np
import matplotlib.pyplot as plt
from model_test import simulate_hematopoiesis, params



## Setup initial parameters
initial_lt_hsc_count = 5000
initial_lt_quiescent = 15000
initial_st_hsc_count = 5
initial_st_quiescent = 5
time_steps = 7 * 40 
sample_days = list(range(0, time_steps, 7))  # If time steps is 7*n, this will produce n samples
seeds = range(100)


# Initialize storage for results
results = []
# Initialize storage for fractions
all_frac_lt_hsc_labeled = []
all_frac_st_wrt_lt_labeled = []

# Run the simulation for each seed
for seed in seeds:
    print(f"Running simulation for seed {seed}...")
    counts, _, _, _, _ = simulate_hematopoiesis(
        initial_lt_hsc_count=initial_lt_hsc_count,
        initial_st_hsc_count=initial_st_hsc_count,
        params=params,
        time_steps=time_steps,
        sample_days=sample_days,
        initial_lt_quiescent=initial_lt_quiescent,
        initial_st_quiescent=initial_st_quiescent,
        seed=seed
    )
    results.append({
        "counts": counts
    })

    # Loop through results to calculate fractions for each seed
    for result in results:
        # Access the counts dictionary within the result
        counts = result["counts"]

        # Calculate fraction of labeled LT-HSCs (including quiescent and active)
        lt_hsc_labeled = np.array(counts["LT-HSC_active_labeled"]) + np.array(counts["LT-HSC_quiescent_labeled"])
        lt_hsc_unlabeled = np.array(counts["LT-HSC_active_unlabeled"]) + np.array(counts["LT-HSC_quiescent_unlabeled"])
        frac_lt_hsc_labeled = lt_hsc_labeled / (lt_hsc_labeled + lt_hsc_unlabeled)
        print(lt_hsc_labeled)

        # Store the fraction of labeled LT-HSCs
        all_frac_lt_hsc_labeled.append(frac_lt_hsc_labeled)

        # Calculate fraction of labeled ST-HSCs (not exp)
        st_hsc_active_labeled = np.array(counts["ST-HSC_active_labeled"]) + np.array(counts["ST-HSC_quiescent_labeled"])
        print(st_hsc_active_labeled)
        # Calculate fraction of labeled ST-HSCs with respect to labeled LT-HSCs
        frac_st_wrt_lt_labeled = st_hsc_active_labeled / lt_hsc_labeled
        # Store the fraction of labeled ST-HSCs with respect to labeled LT-HSCs
        all_frac_st_wrt_lt_labeled.append(frac_st_wrt_lt_labeled)

print(f"all_frac_st_wrt_lt_labeled: {all_frac_st_wrt_lt_labeled}")
# Calculate mean fractions across seeds
mean_frac_lt_hsc_labeled = np.mean(all_frac_lt_hsc_labeled, axis=0)
# Calculate mean fractions of ST-HSCs with respect to LT-HSCs
mean_frac_st_wrt_lt_labeled = np.mean(all_frac_st_wrt_lt_labeled, axis=0)


# Experimental data (fraction of labeled cells in each population)
experimental_data = {
    "LT-HSC": [0.00454015864692873, 0.00970914334678804, 0.0118898059359464, 0.0139842185087899, 0.00935401861275140, 0.0142686401648793, 0.0122944095147182, 0.00952839937711782],
    "ST-HSC": [0.0469708862083901, 0.204464994066137, 0.209602555110320, 0.344848069759038, 0.306764684733494, 0.455197662023728, 0.261024759593610, 0.455403428645269]
    #"MPP": [0.0362029990481008, 0.0526313519252480, 0.0965692203594958, 0.122743713019704, 0.238240948066378, 0.154660386024607, 0.202165606947999, 0.380818537025144]
}
sample_daysexp = [46, 74, 97, 137, 161, 191, 218, 256]

time_points = np.array(sample_days)

# Plot fraction of labeled ST-HSCs with respect to LT-HSCs
plt.figure(figsize=(10, 3))
for frac in all_frac_st_wrt_lt_labeled:
    plt.plot(time_points, frac, color="lightgray", alpha=0.5)  # Each seed in gray
plt.plot(time_points, mean_frac_st_wrt_lt_labeled, color="red", linewidth=2, label="Mean")  # Mean in red
for cell_type, color in zip(["ST-HSC"], ["blue"]):
    plt.plot(sample_daysexp, experimental_data[cell_type], "o", label=f"Experimental {cell_type}", color=color)
# plt.xlabel("Time (days)")
# plt.ylabel("Fraction of Labeled ST-HSCs with respect to LT-HSCs")
# plt.title("Fraction of Labeled ST-HSCs with respect to LT-HSCs Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fraction_st_wrt_lt_labeled.png", dpi=300)
plt.show()

# Plot fraction of labeled LT-HSCs
plt.figure(figsize=(10, 3))
for frac in all_frac_lt_hsc_labeled:
    plt.plot(time_points, frac, color="lightgray", alpha=0.5)  # Each seed in gray
plt.plot(time_points, mean_frac_lt_hsc_labeled, color="red", linewidth=2, label="Mean")  # Mean in red
for cell_type, color in zip(["LT-HSC"], ["blue"]):
    plt.plot(sample_daysexp, experimental_data[cell_type], "o", label=f"Experimental {cell_type}", color=color)
# plt.xlabel("Time (days)")
# plt.ylabel("Fraction of Labeled LT-HSCs")
# plt.title("Fraction of Labeled LT-HSCs Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fraction_labeled_lt_hscs.png", dpi=300)
plt.show()

import pickle

# Save the data used in the figures
data_to_save = {
    "all_frac_lt_hsc_labeled": all_frac_lt_hsc_labeled,
    "all_frac_st_wrt_lt_labeled": all_frac_st_wrt_lt_labeled,
    "mean_frac_lt_hsc_labeled": mean_frac_lt_hsc_labeled,
    "mean_frac_st_wrt_lt_labeled": mean_frac_st_wrt_lt_labeled,
    "time_points": time_points,
    "sample_daysexp": sample_daysexp,
    "experimental_data": experimental_data
}

# Save the data to a pickle file
with open("simulation_results.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

print("Data used in the figures has been saved to 'simulation_results.pkl'.")