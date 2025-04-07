import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns  # For KDE plot
from model_apop import simulate_hematopoiesis

# Define parameters for the simulation
initial_lt_hsc_count = 2 * 250
initial_st_hsc_count = 5 * 2 * 250
initial_lt_quiescent = 5 * 750
initial_st_quiescent = 5 * 2 * 250
time_steps = 7 * 50
sample_days = [0, time_steps]  # Sample at the start and end of the simulation
label_fraction = 0.001
seeds = 500  # Number of seeds to run
paa_values = np.linspace(0.01, 0.1, 2)  # PAA values from 0.01 to 0.1

# Initialize storage for results
results = {}

# Run the simulation for each PAA value
for paa in paa_values:
    print(f"Running simulations for PAA = {paa:.2f}...")
    total_lt_hscs = []  # Store total LT-HSCs for this PAA value

    for seed in range(seeds):
        # Update the PAA value in the parameters
        params = {
            "PAA": paa,  # Probability of apoptosis for LT-HSCs
            "PAB": 0.0025,  # Probability of apoptosis for ST-HSCs
            "PAQA": 0.00,  # Probability of apoptosis for quiescent LT-HSCs
            "PAQB": 0.000,  # Probability of apoptosis for quiescent ST-HSCs
            "LT_HSC_DIVISION_THRESHOLD": 5,
            "ST_HSC_DIVISION_THRESHOLD": 3,
            "PQA": 0.05,
            "PQB": 0.15,
            "P_EXIT_QUIESCENCE_A": 0.05,
            "P_EXIT_QUIESCENCE_B": 0.15,
            "P1A": 0.6,
            "P2A": 0.096,
            "P3A": 0.27968000000000004,
            "P4A": 0.024320000000000005,
            "P1B": 0.5,
            "P2B": 0.15,
            "P3B": 0.1,
            "P4B": 0.25,
            "LT_HSC_TIME_MEAN": 40,
            "LT_HSC_TIME_STD": 12.5,
            "LT_HSC_DAILY_DIVISION_RATE": 0.1,
            "ST_HSC_TIME_MEAN": 12,
            "ST_HSC_TIME_STD": 5.5,
            "ST_HSC_DAILY_DIVISION_RATE": 0.28,
            "LABEL_THRESHOLD": 0.06,
            "LABELING_PERIOD": 2,
            "LABELING_PROB": 0.005,
        }

        # Run the simulation
        counts, _, _, _ = simulate_hematopoiesis(
            initial_lt_hsc_count,
            initial_st_hsc_count,
            params,
            time_steps,
            sample_days,
            label_fraction,
            initial_lt_quiescent,
            initial_st_quiescent,
            seed=seed,
        )

        # Calculate total LT-HSCs at the last time point
        lt_hsc_active = counts["LT-HSC_active_labeled"][-1] + counts["LT-HSC_active_unlabeled"][-1]
        lt_hsc_quiescent = counts["LT-HSC_quiescent_labeled"][-1] + counts["LT-HSC_quiescent_unlabeled"][-1]
        total_lt_hsc = lt_hsc_active + lt_hsc_quiescent
        total_lt_hscs.append(total_lt_hsc)

    # Store the results for this PAA value
    results[paa] = total_lt_hscs

# Save the results to a pickle file
output_file = "lt_hsc_distribution_paa.pkl"
with open(output_file, "wb") as f:
    pickle.dump(results, f)
print(f"Results saved to {output_file}")

# Plot the distribution of total LT-HSCs for each PAA value
for paa, total_lt_hscs in results.items():
    # Calculate min and max values for bins
    min_value = min(total_lt_hscs)
    max_value = max(total_lt_hscs)

    # Use range to create bins for each integer value
    bins = range(min_value, max_value + 2)  # +2 ensures the last value is included

    # Calculate the histogram data
    hist, bin_edges = np.histogram(total_lt_hscs, bins=bins, density=True)  # Normalize the histogram

    # Plot each bar individually
    plt.figure(figsize=(12, 8))
    for i in range(len(hist)):
        # Make the bar for 0 red, others blue
        bar_color = 'red' if bin_edges[i] == 0 else 'blue'
        # Center the bar at the midpoint of the bin
        bar_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        plt.bar(bar_center, hist[i], width=1, color=bar_color, edgecolor='black', align='center', alpha=0.7)

    # Add KDE line using seaborn
    sns.kdeplot(total_lt_hscs, color='green', linewidth=2, label='KDE Fit')

    # Add labels and title
    plt.xlabel("Total LT-HSCs")
    plt.ylabel("Density")
    plt.title(f"Distribution of Active + Quiescent LT-HSC - PAA = {paa:.2f}")
    plt.grid(axis='y', alpha=0.75)
    plt.legend()

    # Save the figure
    plt.savefig(f"quiescent_active_lt_hsc_distribution_PAA_{paa:.2f}.png", format="png", dpi=300)

    # Show the plot
    plt.show()
