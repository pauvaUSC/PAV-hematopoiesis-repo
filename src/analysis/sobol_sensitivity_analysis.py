import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from hematopoiesis_model_v2 import simulate_hematopoiesis, params

# Define the Sobol problem
problem = {
    "num_vars": 11,
    "names": [
        "LT_HSC_DIVISION_THRESHOLD", "ST_HSC_DIVISION_THRESHOLD",
        "P2A", "P3A", "P4A",  # LT-HSC division probabilities (P1A is fixed)
        "P2B", "P3B", "P4B",  # ST-HSC division probabilities (P1B is fixed)
        "LABEL_THRESHOLD", "LABELING_PERIOD", "LABELING_PROB"
    ],
    "bounds": [
        [3, 7],  # LT_HSC_DIVISION_THRESHOLD
        [2, 5],  # ST_HSC_DIVISION_THRESHOLD
        [0.07, 0.2],  # P2A (Asymmetric Division for LT-HSCs)
        [0.02, 0.07],  # P3A (Symmetric Differentiation for LT-HSCs)
        [0.05, 0.14],  # P4A (Full Differentiation for LT-HSCs)
        [0.1, 0.2],  # P2B (Asymmetric Division for ST-HSCs)
        [0.1, 0.175],  # P3B (Symmetric Differentiation for ST-HSCs)
        [0.125, 0.2],  # P4B (Full Differentiation for ST-HSCs)
        [0.02, 0.1],  # LABEL_THRESHOLD
        [1, 3],  # LABELING_PERIOD
        [0.01, 0.1],  # LABELING_PROB
    ]
}

# Generate Sobol samples
param_values = saltelli.sample(problem, 1024, calc_second_order=True)

# Initialize storage for results
results = []

# Run the simulations for each parameter set
for i, param_set in enumerate(param_values):
    # Update parameters
    params["LT_HSC_DIVISION_THRESHOLD"] = int(param_set[0])
    params["ST_HSC_DIVISION_THRESHOLD"] = int(param_set[1])
    params["P2A"] = param_set[2]
    params["P3A"] = param_set[3]
    params["P4A"] = 0.4 - param_set[2] - param_set[3]  # Ensure sum to 1 with P1A=0.6
    params["P2B"] = param_set[4]
    params["P3B"] = param_set[5]
    params["P4B"] = 0.5 - param_set[4] - param_set[5]  # Ensure sum to 1 with P1B=0.5
    params["LABEL_THRESHOLD"] = param_set[6]
    params["LABELING_PERIOD"] = int(param_set[7])
    params["LABELING_PROB"] = param_set[8]

    # Run the simulation
    initial_lt_hsc_count = 2*250
    initial_st_hsc_count = 3*2*250
    initial_lt_quiescent = 2*750
    initial_st_quiescent = 3*2*250
    time_steps = 7 * 50
    sample_days = list(range(0, time_steps, 7))  # If time steps is 7*n, this will produce n samples
    label_fraction = 0.001
    seed = 42

    counts, _, _, _ = simulate_hematopoiesis(
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

    # Store the result (e.g., total LT-HSC active counts at the last time step)
    LT_HSC_active_labeled = counts["LT-HSC_active_labeled"][-1]
    LT_HSC_active_unlabeled = counts["LT-HSC_active_unlabeled"][-1]
    LT_HSC_active = LT_HSC_active_labeled + LT_HSC_active_unlabeled
    results.append(LT_HSC_active)

# Perform Sobol analysis
sobol_indices = sobol.analyze(problem, np.array(results), calc_second_order=True)

# Plot Sobol indices
def plot_sobol_indices(sobol_indices, title, filename):
    # Extract first-order and total-order indices
    first_order = sobol_indices["S1"]
    total_order = sobol_indices["ST"]

    # Plot the indices
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    x = np.arange(len(problem["names"]))

    plt.bar(x - bar_width / 2, first_order, bar_width, label="First-order Sobol Index")
    plt.bar(x + bar_width / 2, total_order, bar_width, label="Total-order Sobol Index")

    plt.xticks(x, problem["names"], rotation=45, ha="right")
    plt.ylabel("Sobol Index")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename, format="png", dpi=300)
    plt.close()

# Generate and save the Sobol index plot
plot_sobol_indices(sobol_indices, "Sobol Sensitivity Analysis", "sobol_sensitivity_analysis.png")
