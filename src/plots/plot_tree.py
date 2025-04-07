# This file generates tree plots for the hematopoiesis model.
# It includes functions to create and customize plots.

import matplotlib.pyplot as plt
import networkx as nx

def create_tree_plot(tree_data, title="Hematopoiesis Model Tree Plot"):
    """
    Generates a tree plot from the provided tree data.

    Parameters:
    tree_data (dict): A dictionary representing the tree structure.
    title (str): The title of the plot.

    Returns:
    None: Displays the plot.
    """
    # Create a directed graph from the tree data
    G = nx.DiGraph(tree_data)

    # Set the figure size
    plt.figure(figsize=(10, 8))

    # Draw the tree using a spring layout
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')

    # Set the title of the plot
    plt.title(title)

    # Show the plot
    plt.show()

def main():
    """
    Main function to execute the tree plot generation.
    
    This function can be modified to include specific tree data for visualization.
    """
    # Example tree data (to be replaced with actual model data)
    example_tree_data = {
        'Root': ['Child1', 'Child2'],
        'Child1': ['Grandchild1', 'Grandchild2'],
        'Child2': ['Grandchild3']
    }

    # Create the tree plot
    create_tree_plot(example_tree_data)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()