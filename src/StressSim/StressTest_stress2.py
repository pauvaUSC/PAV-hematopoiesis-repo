import pickle
import numpy as np
from hematopoiesis_model_v5 import simulate_hematopoiesis, params

#######

import pickle
import numpy as np
from hematopoiesis_model_v5 import simulate_hematopoiesis, params

params["PAA"] = 0.01
params["PAB"] = 0.01
params["P_EXIT_QUIESCENCE_B"] = 0.7
params["PQB"] = 0.7


# Define parameters for the simulation
initial_lt_hsc_count = 500
initial_st_hsc_count = 1500
initial_lt_quiescent = 5000
initial_st_quiescent = 5000
time_steps = 7 * 100  
sample_days = list(range(0, time_steps, 7))  # Sample every 7 days
num_seeds = 100  # Number of seeds to run

# Initialize storage for results
all_counts = []
all_cells_history = []
all_division_events = []
all_apoptosis_events = []

# Run the simulation for multiple seeds
for seed in range(num_seeds):
    print(f"Running simulation for seed {seed + 1}/{num_seeds}...")
    counts, cells, cells_history, division_events, apoptosis_events = simulate_hematopoiesis(
        initial_lt_hsc_count,
        initial_st_hsc_count,
        params,
        time_steps,
        sample_days,
        initial_lt_quiescent,
        initial_st_quiescent,
        seed=seed,
    )

    # Store the results for this seed
    all_counts.append(counts)
    all_cells_history.append(cells_history)
    all_division_events.append(division_events)
    all_apoptosis_events.append(apoptosis_events)

# Save all the data to a pickle file
output_file = "hematopoiesis_simulation_data_stress2.pkl"
with open(output_file, "wb") as f:
    pickle.dump({
        "counts": all_counts,
        "cells_history": all_cells_history,
        "division_events": all_division_events,
        "apoptosis_events": all_apoptosis_events,
    }, f)

print(f"Simulation data saved to {output_file}")