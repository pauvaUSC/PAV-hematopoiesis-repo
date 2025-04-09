import numpy as np
import matplotlib.pyplot as plt
from hematopoiesis_model_v2 import simulate_hematopoiesis, params

# Simulation parameters
initial_lt_hsc_count = 2 * 250
initial_st_hsc_count = 5 * 2 * 250
initial_lt_quiescent = 2 * 750
initial_st_quiescent = 5 * 2 * 250
time_steps = 7 * 50
sample_days = list(range(0, time_steps, 7))  # If time steps is 7*n, this will produce n samples
label_fraction = 0.001
seeds = range(100)

# Initialize storage for results
results = []
# Initialize storage for fractions
all_frac_lt_hsc_labeled = []
all_frac_st_wrt_lt_labeled = []

# Run the simulation for each seed
for seed in seeds:
    print(f"Running simulation for seed {seed}...")
    counts, cells_history, division_events, apoptosis_events = simulate_hematopoiesis(
        initial_lt_hsc_count,
        initial_st_hsc_count,
        params,
        time_steps,
        sample_days,
        label_fraction,
        initial_lt_quiescent,
        initial_st_quiescent,
        seed=seed
    )
    results.append({
        "counts": counts,
        "cells_history": cells_history,
        "division_events": division_events,
        "apoptosis_event": apoptosis_events
    })

    # Loop through results to calculate fractions for each seed
    for result in results:
        # Access the counts dictionary within the result
        counts = result["counts"]

        # Calculate fraction of labeled LT-HSCs (including quiescent and active)
        lt_hsc_labeled = np.array(counts["LT-HSC_active_labeled"]) + np.array(counts["LT-HSC_quiescent_labeled"])
        lt_hsc_unlabeled = np.array(counts["LT-HSC_active_unlabeled"]) + np.array(counts["LT-HSC_quiescent_unlabeled"])
        frac_lt_hsc_labeled = lt_hsc_labeled / (lt_hsc_labeled + lt_hsc_unlabeled)

        # Store the fraction of labeled LT-HSCs
        all_frac_lt_hsc_labeled.append(frac_lt_hsc_labeled)

        # Calculate fraction of labeled ST-HSCs (not exp)
        st_hsc_active_labeled = np.array(counts["ST-HSC_active_labeled"]) + np.array(counts["ST-HSC_quiescent_labeled"])
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
sample_daysexp = [16, 44, 67, 107, 131, 161, 188, 226]
sample_daysexp = [day + 15 for day in sample_daysexp] # move 15 days to let simulation stabilize before comparing to exp

time_points = np.array(sample_days)

# Plot fraction of labeled ST-HSCs with respect to LT-HSCs
plt.figure(figsize=(10, 6))
for frac in all_frac_st_wrt_lt_labeled:
    plt.plot(time_points, frac, color="gray", alpha=0.5)  # Each seed in gray
plt.plot(time_points, mean_frac_st_wrt_lt_labeled, color="red", linewidth=2, label="Mean")  # Mean in red
for cell_type, color in zip(["ST-HSC"], ["blue"]):
    plt.plot(sample_daysexp, experimental_data[cell_type], "o", label=f"Experimental {cell_type}", color=color)
plt.xlabel("Time (days)")
plt.ylabel("Fraction of Labeled ST-HSCs with respect to LT-HSCs")
plt.title("Fraction of Labeled ST-HSCs with respect to LT-HSCs Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fraction_st_wrt_lt_labeled.png", dpi=300)
plt.show()

# Plot fraction of labeled LT-HSCs
plt.figure(figsize=(10, 6))
for frac in all_frac_lt_hsc_labeled:
    plt.plot(time_points, frac, color="lightgray", alpha=0.5)  # Each seed in gray
plt.plot(time_points, mean_frac_lt_hsc_labeled, color="red", linewidth=2, label="Mean")  # Mean in red
for cell_type, color in zip(["LT-HSC"], ["blue"]):
    plt.plot(sample_daysexp, experimental_data[cell_type], "o", label=f"Experimental {cell_type}", color=color)
plt.xlabel("Time (days)")
plt.ylabel("Fraction of Labeled LT-HSCs")
plt.title("Fraction of Labeled LT-HSCs Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fraction_labeled_lt_hscs.png", dpi=300)
plt.show()

# # Plot labeled and unlabeled LT-HSCs
# plt.figure(figsize=(10, 6))
# plt.plot(time_points, mean_lt_hsc_labeled, color="blue", label="Labeled LT-HSCs")
# plt.plot(time_points, mean_lt_hsc_unlabeled, color="orange", label="Unlabeled LT-HSCs")
# plt.xlabel("Time (days)")
# plt.ylabel("Count")
# plt.title("Labeled and Unlabeled LT-HSCs Over Time")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("lt_hscs_labeled_unlabeled.png", dpi=300)

# # Plot fraction of labeled ST-HSCs
# plt.figure(figsize=(10, 6))
# for frac in all_frac_st_hsc_active_labeled:
#     plt.plot(time_points, frac, color="gray", alpha=0.5)  # Each seed in gray
# plt.plot(time_points, mean_frac_st_hsc_active_labeled, color="red", linewidth=2, label="Mean")  # Mean in red
# plt.xlabel("Time (days)")
# plt.ylabel("Fraction of Labeled ST-HSCs")
# plt.title("Fraction of Labeled ST-HSCs Over Time")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("fraction_labeled_st_hscs.png", dpi=300)
# plt.show()

# # Plot labeled and unlabeled ST-HSCs
# plt.figure(figsize=(10, 6))
# plt.plot(time_points, mean_st_hsc_active_labeled, color="blue", label="Labeled ST-HSCs")
# plt.plot(time_points, mean_st_hsc_active_unlabeled, color="orange", label="Unlabeled ST-HSCs")
# plt.xlabel("Time (days)")
# plt.ylabel("Count")
# plt.title("Labeled and Unlabeled ST-HSCs Over Time")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("st_hscs_labeled_unlabeled.png", dpi=300)
# plt.show()


import pickle

# Save all outputs to a file
output_data = {
    "counts": counts,
    "cells_history": cells_history,
    "division_events": division_events,
    "apoptosis_event": apoptosis_events
}

import pickle

# Save all outputs for all seeds and parameters to a file
output_data = {
    "parameters": {
        "initial_lt_hsc_count": initial_lt_hsc_count,
        "initial_st_hsc_count": initial_st_hsc_count,
        "initial_lt_quiescent": initial_lt_quiescent,
        "initial_st_quiescent": initial_st_quiescent,
        "time_steps": time_steps,
        "sample_days": sample_days,
        "label_fraction": label_fraction,
        "seeds": list(seeds),
        "params": params  # Save the simulation parameters
    },
    "results": {
        "all_frac_lt_hsc_labeled": [frac.tolist() for frac in all_frac_lt_hsc_labeled],
        "all_frac_st_wrt_lt_labeled": [frac.tolist() for frac in all_frac_st_wrt_lt_labeled],
        "mean_frac_lt_hsc_labeled": mean_frac_lt_hsc_labeled.tolist(),
        "mean_frac_st_wrt_lt_labeled": mean_frac_st_wrt_lt_labeled.tolist()
    },
    "all_outputs": results  # Save all outputs for all seeds
}

# Write the data to a pickle file
output_file = "simulation_outputs_with_seeds.pkl"
with open(output_file, "wb") as f:
    pickle.dump(output_data, f)



print(f"Simulation outputs for all seeds and parameters saved to {output_file}")

# import pickle

# # Load the simulation outputs
# with open("simulation_outputs_with_seeds.pkl", "rb") as f:
#     loaded_data = pickle.load(f)

# # Access the saved parameters and results
# parameters = loaded_data["parameters"]
# results = loaded_data["results"]
# all_outputs = loaded_data["all_outputs"]

# print("Simulation outputs for all seeds and parameters loaded successfully!")
