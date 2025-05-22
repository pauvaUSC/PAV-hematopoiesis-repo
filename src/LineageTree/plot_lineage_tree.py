import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

import os
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

from hematopoiesis_model_v5 import simulate_hematopoiesis, params

# Filepath for the simulation data
pkl_file = "simulation_data.pkl"

initial_lt_hsc_count = 500
initial_st_hsc_count = 1500
initial_lt_quiescent = 5000
initial_st_quiescent = 5000
time_steps = 7 * 50
sample_days = [time_steps]  # If time steps is 7*n, this will produce n samples
seed = 42

# Check if the .pkl file exists
if os.path.exists(pkl_file):
    print(f"Loading data from {pkl_file}...")
    with open(pkl_file, "rb") as f:
        counts, cells_history, division_events, apoptosis_events, cells = pickle.load(f)
else:
    print("Running simulation...")
    # Run the simulation
    initial_lt_hsc_count = initial_lt_hsc_count
    initial_st_hsc_count = initial_st_hsc_count
    initial_lt_quiescent = initial_lt_quiescent
    initial_st_quiescent = initial_st_quiescent
    time_steps = time_steps
    sample_days = [time_steps]  
    seed = seed

    counts, cells, cells_history, division_events, apoptosis_events = simulate_hematopoiesis(
        initial_lt_hsc_count,
        initial_st_hsc_count,
        params,
        time_steps,
        sample_days,
        initial_lt_quiescent,
        initial_st_quiescent,
        seed=seed
    )

    # Save the simulation data to a .pkl file
    with open(pkl_file, "wb") as f:
        pickle.dump((counts, cells_history, division_events, apoptosis_events, cells), f)
    print(f"Simulation data saved to {pkl_file}.")


# Pick a random initial LT-HSC
initial_cell_id = 4764

# Trace the lineage of the selected cell, including active and quiescent descendants
def trace_lineage(cell_id, cells, cells_history):
    """
    Trace the lineage of a given cell, including active, quiescent, and removed descendants.
    Args:
        cell_id (int): The unique ID of the cell to trace.
        cells (list): The current population of cells.
        cells_history (list): The history of removed cells.
    Returns:
        list: A list of cells in the lineage, including the selected cell and all its descendants.
    """
    lineage = []
    stack = [cell_id]  # Start with the selected cell ID

    # Combine current cells and cells_history for full lineage tracking
    all_cells = [
        {
            "unique_id": cell.unique_id,
            "parent_id": cell.parent_id,
            "cell_type": cell.cell_type,
            "state": cell.state,
        }
        for cell in cells
    ] + [
        {
            "unique_id": c["unique_id"],
            "parent_id": c["parent_id"],
            "cell_type": c["cell_type"],
            "state": c.get("state", "Removed"),  # Use a default value if "state" is missing
        }
        for c in cells_history
    ]

    # Use a stack to perform depth-first search for descendants
    while stack:
        current_id = stack.pop()
        cell = next((c for c in all_cells if c["unique_id"] == current_id), None)
        if cell:
            lineage.append(cell)
            # Add all children of the current cell to the stack
            children = [c["unique_id"] for c in all_cells if c["parent_id"] == current_id]
            stack.extend(children)

    return lineage

# Trace the lineage of the selected cell
lineage = trace_lineage(initial_cell_id, cells, cells_history)


def plot_hierarchical_tree_with_progress(lineage, initial_cell_id,time_steps):
    """
    Plot a hierarchical tree of the lineage with the initial cell at the top.
    Includes a progress counter during tree generation.
    Args:
        lineage (list): List of cells in the lineage.
        initial_cell_id (int): The unique ID of the initial cell.
    """
    import time

    G = nx.DiGraph()
    color_map = []
    node_shapes = []
    node_edge_colors = []

    # Add nodes and edges to the graph
    print("Generating hierarchical tree...")
    for i, cell in enumerate(lineage):
        G.add_node(cell["unique_id"], cell_type=cell["cell_type"])
        if cell["parent_id"] is not None:
            G.add_edge(cell["parent_id"], cell["unique_id"])

        # Assign colors and shapes based on cell type and state
        if cell["unique_id"] == initial_cell_id:
            node_edge_colors.append("black") 
            if cell.get("state") == "Non-quiescent Active":
                color_map.append("blue")
                node_shapes.append("o")
            elif cell.get("state") == "Non-quiescent Inactive":
                color_map.append("dodgerblue")
                node_shapes.append("d")
            elif cell.get("state") == "Quiescent":
                color_map.append("skyblue")
                node_shapes.append("o")
            elif cell.get("state") == "Removed":
                color_map.append("lightblue")
                node_shapes.append("s")
        elif cell["cell_type"] == "LT-HSC":
            node_edge_colors.append("white") 
            if cell.get("state") == "Non-quiescent Active":
                color_map.append("blue")
                node_shapes.append("o")
            elif cell.get("state") == "Non-quiescent Inactive":
                color_map.append("dodgerblue")
                node_shapes.append("d")
            elif cell.get("state") == "Quiescent":
                color_map.append("skyblue")
                node_shapes.append("o")
            elif cell.get("state") == "Removed":
                color_map.append("lightblue")
                node_shapes.append("s")
        elif cell["cell_type"] == "ST-HSC":
            node_edge_colors.append("white") 
            if cell.get("state") == "Non-quiescent Active":
                node_shapes.append("o")
                color_map.append("green")
            elif cell.get("state") == "Non-quiescent Inactive":
                color_map.append("limegreen")
                node_shapes.append("d")
            elif cell.get("state") == "Quiescent":
                color_map.append("palegreen")
                node_shapes.append("o")
            elif cell.get("state") == "Removed":
                color_map.append("lightgreen")
                node_shapes.append("s")
        elif cell["cell_type"] == "MPP":
            node_edge_colors.append("white") 
            if cell.get("state") == "Non-quiescent Active":
                color_map.append("red")
                node_shapes.append("o")
            elif cell.get("state") == "Non-quiescent Inactive":
                color_map.append("orangered")
                node_shapes.append("d")
            elif cell.get("state") == "Quiescent":
                color_map.append("salmon")
                node_shapes.append("o")
            elif cell.get("state") == "Removed":
                color_map.append("lightcoral")
                node_shapes.append("s")

        # Print progress every 100 nodes
        if (i + 1) % 100 == 0 or i == len(lineage) - 1:
            print(f"Processed {i + 1}/{len(lineage)} nodes...")
            time.sleep(0.01)  # Simulate processing delay for visibility

    # Use a hierarchical layout (requires pygraphviz)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # 'dot' creates a hierarchical layout

    # Plot the graph
    plt.figure(figsize=(12, 8))

    # Draw nodes with different shapes
    for shape in set(node_shapes):  # Iterate over unique shapes
        shape_nodes = [node for node, s in zip(G.nodes(), node_shapes) if s == shape]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=shape_nodes,
            node_color=[color_map[i] for i, node in enumerate(G.nodes()) if node in shape_nodes],
            node_shape=shape,
            node_size=800,
            edgecolors=[node_edge_colors[i] for i, node in enumerate(G.nodes()) if node in shape_nodes],
            linewidths=2,  # Set the border width
        )

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # Add a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='LT-HSC Active'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='dodgerblue', markersize=10, label='LT-HSC Senescent'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='LT-HSC Quiescent'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', markersize=10, label='LT-HSC Removed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='ST-HSC Active'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='limegreen', markersize=10, label='ST-HSC Senescent'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='palegreen', markersize=10, label='ST-HSC Quiescent'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', markersize=10, label='ST-HSC Removed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='MPP Active'),
        # Line2D([0], [0], marker='d', color='w', markerfacecolor='orangered', markersize=10, label='MPP Senescent'),
        # Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=10, label='MPP Quiescent'),
        # Line2D([0], [0], marker='s', color='w', markerfacecolor='lightcoral', markersize=10, label='MPP Removed'),
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.title(f"Hierarchical Lineage Tree of Cell ID {initial_cell_id} after {time_steps} days")

plot_hierarchical_tree_with_progress(lineage, initial_cell_id,time_steps)


# Save the figure
plt.savefig(f"hierarchical_tree_cell_{initial_cell_id}.png", format="png", dpi=300)
plt.show()

