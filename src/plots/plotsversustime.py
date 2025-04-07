# plotsversustime.py

"""
This module contains functions to create plots that visualize model outputs over time.
It includes various time-series plots to analyze the dynamics of hematopoiesis.
"""

import matplotlib.pyplot as plt

def plot_time_series(data, title='Model Outputs Over Time', xlabel='Time', ylabel='Output'):
    """
    Plots a time series graph for the given data.

    Parameters:
    data (dict): A dictionary where keys are time points and values are the corresponding outputs.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data.keys(), data.values(), marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_multiple_time_series(data_dict, title='Model Outputs Over Time', xlabel='Time', ylabel='Output'):
    """
    Plots multiple time series graphs for the given data.

    Parameters:
    data_dict (dict): A dictionary where keys are labels for each series and values are dictionaries
                      with time points as keys and corresponding outputs as values.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    for label, data in data_dict.items():
        plt.plot(data.keys(), data.values(), marker='o', label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()