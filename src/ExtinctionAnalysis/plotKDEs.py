import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Uncomment if you want to save results

from hematopoiesis_model_v5 import simulate_hematopoiesis, params

def run_and_plot_kde():
    initial_lt_hsc_count = 250
    initial_st_hsc_count = 750
    initial_lt_quiescent = 5
    initial_st_quiescent = 5
    time_steps = 5 * 40
    sample_days = list(range(0, time_steps + 1, 5))  
    num_seeds = 500

    paa_values = [0.03, 0.04, 0.05, 0.06, 0.07]
    results = {}

    plt.figure(figsize=(12, 8))

    for paa in paa_values:
        params["PAA"] = paa
        #params["PAQA"] = paa
        active_lt_hsc_values = []

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

            active_lt_hsc = np.array(counts["LT-HSC_active_labeled"]) + np.array(counts["LT-HSC_active_unlabeled"])
            # # Check if lists are not empty before accessing the last element
            # if counts["LT-HSC_active_labeled"] and counts["LT-HSC_active_unlabeled"]:
            #     active_lt_hsc = counts["LT-HSC_active_labeled"][-2] + counts["LT-HSC_active_unlabeled"][-2]
            # else:
            #     active_lt_hsc = 0  
            active_lt_hsc_values.append(active_lt_hsc[-1])

        results[paa] = active_lt_hsc_values

        # Save individual PAA data to a pickle file
        paasave = str(paa).replace('.', '')
        with open(f"lt_hsc_distribution_paa_{paasave}.pkl", "wb") as f:
            pickle.dump({paa: active_lt_hsc_values}, f)

        if not active_lt_hsc_values:
            print(f"No data to plot for PAA {paa}. Skipping...")
            continue

        if not active_lt_hsc_values:
            print(f"No data to plot for PAA {paa}. Skipping...")
            continue

        kde = sns.kdeplot(active_lt_hsc_values, label=f"PAA = {paa:.2f}", linewidth=2)
        x, y = kde.get_lines()[-1].get_data()
        plt.fill_between(x, y, alpha=0.3)

    # plt.xlabel("Total LT-HSCs")
    # plt.ylabel("Density")
    # plt.title("KDEs of Total LT-HSCs for Different PAA Values")
    # plt.grid(axis='y', alpha=0.75)
    # plt.legend()
    # plt.savefig("kde_lt_hsc_paa_values.png", format="png", dpi=300)

    plt.xlabel("Total LT-HSCs")
    plt.ylabel("Density")
    plt.title("KDEs of Total LT-HSCs for Different PAA Values")
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.xlim(left=0)  # <-- Add this line to hide negative x values
    plt.savefig("kde_lt_hsc_paa_values.png", format="pdf", dpi=300)
    

    plt.show()

    # Optional: Save results for later use
    with open("lt_hsc_distribution_results.pkl", "wb") as f:
        pickle.dump(results, f)

run_and_plot_kde()