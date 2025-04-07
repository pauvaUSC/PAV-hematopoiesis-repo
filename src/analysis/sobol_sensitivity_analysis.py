# This file implements Sobol sensitivity analysis for the hematopoiesis model.
# It includes functions to calculate sensitivity indices and visualize results.

import numpy as np
import matplotlib.pyplot as plt
from SALib.analyze import sobol
from SALib.sample import saltelli

def run_sobol_analysis(model, params, num_samples=1000):
    """
    Run Sobol sensitivity analysis on the given model.

    Parameters:
    model: function
        The hematopoiesis model function to analyze.
    params: dict
        A dictionary containing parameter names and their bounds.
    num_samples: int
        The number of samples to generate for the analysis.

    Returns:
    dict
        A dictionary containing the Sobol sensitivity indices.
    """
    # Generate samples using Saltelli's sampling method
    param_names = list(params.keys())
    bounds = np.array(list(params.values()))
    samples = saltelli.sample(bounds, num_samples)

    # Run the model for each sample and collect results
    model_outputs = np.array([model(*sample) for sample in samples])

    # Perform Sobol analysis
    Si = sobol.analyze(params, model_outputs)

    return Si

def visualize_sobol_indices(Si):
    """
    Visualize the Sobol sensitivity indices.

    Parameters:
    Si: dict
        A dictionary containing the Sobol sensitivity indices.
    """
    # Extract first-order and total indices
    first_order = Si['S1']
    total_order = Si['ST']

    # Create bar plots for first-order and total indices
    indices = np.arange(len(first_order))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(indices, first_order, width, label='First Order')
    bars2 = ax.bar(indices + width, total_order, width, label='Total Order')

    # Add labels and title
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Sensitivity Indices')
    ax.set_title('Sobol Sensitivity Analysis')
    ax.set_xticks(indices + width / 2)
    ax.set_xticklabels(Si['names'], rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()