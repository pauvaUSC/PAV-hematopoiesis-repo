from skopt import gp_minimize
from skopt.space import Real
from model_test import simulate_hematopoiesis, params
import numpy as np

# Experimental data
experimental_data = {
    "ST-HSC": [0.0469708862083901, 0.204464994066137, 0.209602555110320, 0.344848069759038, 
               0.306764684733494, 0.455197662023728, 0.261024759593610, 0.455403428645269]
}
sample_daysexp = [46, 74, 97, 137, 161, 191, 218, 256]

# Simulation parameters
time_steps = sample_daysexp[-1] + 5
sample_days = sample_daysexp
num_seeds = 5  # Number of seeds for stochastic runs

# Define the search space for the parameters to fit
search_space = [
    Real(0.01, 0.5, name="P2A"),
    Real(0.01, 0.5, name="P3A"),
    Real(0.01, 0.4, name="P2B"),
    Real(0.01, 0.4, name="P3B"),
    Real(2, 20, name="LT_HSC_DIVISION_THRESHOLD"),
    Real(2, 20, name="ST_HSC_DIVISION_THRESHOLD"),
    Real(5, 20, name="LT_HSC_TIME_STD"),
    Real(2, 10, name="ST_HSC_TIME_STD"),
]

# Parameters to fit
fit_params = [
    "P2A", "P3A", "P2B", "P3B",
    "LT_HSC_DIVISION_THRESHOLD", "ST_HSC_DIVISION_THRESHOLD",
    "LT_HSC_TIME_STD", "ST_HSC_TIME_STD"
]

# Objective function to minimize
def objective_function(fit_values):
    # Update the parameters with the current fit values
    for i, param in enumerate(fit_params):
        params[param] = fit_values[i]

    # Calculate dependent probabilities
    params["P4A"] = 1 - params["P1A"] - params["P2A"] - params["P3A"]
    params["P4B"] = 1 - params["P1B"] - params["P2B"] - params["P3B"]

    # Reject invalid parameter sets
    if params["P4A"] < 0 or params["P4B"] < 0:
        return 1e6  # Penalize invalid parameter sets
    if not (0 <= params["P4A"] <= 1) or not (0 <= params["P4B"] <= 1):
        return 1e6
    if not (np.isclose(params["P1A"] + params["P2A"] + params["P3A"] + params["P4A"], 1)):
        return 1e6
    if not (np.isclose(params["P1B"] + params["P2B"] + params["P3B"] + params["P4B"], 1)):
        return 1e6

    # Run the simulation for multiple seeds and calculate the mean fractions
    st_hsc_fractions = []
    for seed in range(num_seeds):
        try:
            counts, _, _, _, _ = simulate_hematopoiesis(
                initial_lt_hsc_count=5000,
                initial_st_hsc_count=5,
                params=params,
                time_steps=time_steps,
                sample_days=sample_days,
                initial_lt_quiescent=15000,
                initial_st_quiescent=5,
                seed=seed
            )
        except Exception as e:
            print(f"Simulation failed with error: {e}")
            return 1e6  # Penalize invalid parameter sets

        # Calculate fractions
        lt_hsc_labeled = np.array(counts["LT-HSC_active_labeled"]) + np.array(counts["LT-HSC_quiescent_labeled"])
        st_hsc_labeled = np.array(counts["ST-HSC_active_labeled"]) + np.array(counts["ST-HSC_quiescent_labeled"])
        frac_st_hsc_labeled = st_hsc_labeled / lt_hsc_labeled
        st_hsc_fractions.append(frac_st_hsc_labeled)

    # Calculate the mean fractions across seeds
    mean_st_hsc_fractions = np.mean(st_hsc_fractions, axis=0)

    # Calculate the error (sum of squared differences)
    error_st = np.sum((mean_st_hsc_fractions - experimental_data["ST-HSC"])**2)
    return error_st

# Perform Bayesian optimization
result = gp_minimize(
    objective_function,  # Objective function
    search_space,        # Parameter search space
    n_calls=100,          # Number of iterations
    random_state=42,     # For reproducibility
    verbose=True         # Print progress during optimization
)

# Output the best-fit parameters
print("Best parameters:")
for i, param in enumerate(fit_params):
    print(f"{param}: {result.x[i]}")
print("Best error:", result.fun)
