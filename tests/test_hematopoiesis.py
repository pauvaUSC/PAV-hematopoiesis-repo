# test_hematopoiesis.py

"""
Unit tests for the hematopoiesis model and analysis scripts.

This module contains test cases to ensure that the functions and classes
in the hematopoiesis model and analysis scripts behave as expected.
Each test is designed to validate specific functionality and output.
"""

import unittest
from src.models.hematopoiesis_model_v2 import HematopoiesisModel
from src.analysis.extinction_analysis import analyze_extinction
from src.analysis.sobol_sensitivity_analysis import calculate_sensitivity_indices
from src.plots.plot_tree import generate_tree_plot
from src.plots.plotsversustime import create_time_series_plots

class TestHematopoiesisModel(unittest.TestCase):
    
    def setUp(self):
        """Set up the HematopoiesisModel instance for testing."""
        self.model = HematopoiesisModel(parameters={'param1': 1, 'param2': 2})
    
    def test_model_initialization(self):
        """Test the initialization of the hematopoiesis model."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.parameters['param1'], 1)
    
    def test_extinction_analysis(self):
        """Test the extinction analysis function."""
        result = analyze_extinction(data=[1, 2, 3])
        self.assertIsInstance(result, dict)  # Assuming the result is a dictionary
    
    def test_sensitivity_analysis(self):
        """Test the Sobol sensitivity analysis function."""
        indices = calculate_sensitivity_indices(model=self.model)
        self.assertGreater(len(indices), 0)  # Ensure indices are calculated
    
    def test_generate_tree_plot(self):
        """Test the tree plot generation function."""
        plot = generate_tree_plot(data=self.model.get_tree_data())
        self.assertIsNotNone(plot)  # Ensure a plot is generated
    
    def test_create_time_series_plots(self):
        """Test the time series plot creation function."""
        plots = create_time_series_plots(model=self.model)
        self.assertGreater(len(plots), 0)  # Ensure plots are created

if __name__ == '__main__':
    unittest.main()