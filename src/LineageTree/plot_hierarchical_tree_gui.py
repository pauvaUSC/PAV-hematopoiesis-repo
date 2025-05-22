import os
import pickle
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Import Line2D for legend elements
import networkx as nx
from hematopoiesis_model_v5 import simulate_hematopoiesis, params

# Filepath for the simulation data
pkl_file = "simulation_data.pkl"



# Function to trace the lineage of a cell
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

# Function to plot the hierarchical tree
def plot_hierarchical_tree_with_progress(lineage, initial_cell_id, time_steps, canvas, figure):
    """
    Plot a hierarchical tree of the lineage with the initial cell at the top.
    Includes a progress counter during tree generation.
    Args:
        lineage (list): List of cells in the lineage.
        initial_cell_id (int): The unique ID of the initial cell.
        time_steps (int): The number of time steps in the simulation.
        canvas (FigureCanvasTkAgg): The canvas to draw the plot on.
        figure (matplotlib.figure.Figure): The figure to use for the plot.
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

    # Clear the previous plot
    figure.clear()

    # Plot the graph
    ax = figure.add_subplot(111)
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
            ax=ax,
        )

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, edge_color="gray", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)

    # Add a title
    ax.set_title(f"Hierarchical Lineage Tree of Cell ID {initial_cell_id} after {time_steps} days")

    # Refresh the canvas
    canvas.draw()

# GUI Functionality
def run_gui():
    # Load or run the simulation
    global counts, cells_history, division_events, apoptosis_events, cells

    print(f"Loading data from {pkl_file}...")
    with open(pkl_file, "rb") as f:
        counts, cells_history, division_events, apoptosis_events, cells = pickle.load(f)
    
    

    # Create the main window
    root = tk.Tk()
    root.title("Lineage Tree Viewer")

    # Create a frame for the controls
    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

    # Create a label and entry for the cell ID
    tk.Label(control_frame, text="Cell ID:").pack(side=tk.LEFT, padx=5)
    cell_id_entry = tk.Entry(control_frame, width=10)
    cell_id_entry.pack(side=tk.LEFT, padx=5)
    cell_id_entry.insert(0, "47")  # Default value

    # Create a button to plot the lineage tree
    plot_button = tk.Button(control_frame, text="Plot Lineage Tree")
    plot_button.pack(side=tk.LEFT, padx=5)

    # Create a frame for the plot
    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Create a matplotlib figure and canvas
    figure = plt.Figure(figsize=(8, 6))
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Function to handle the "Plot Lineage Tree" button
    def plot_tree():
        try:
            # Get the cell ID from the entry
            initial_cell_id = int(cell_id_entry.get())

            time_steps = 7 * 50
            # Trace the lineage
            lineage = trace_lineage(initial_cell_id, cells, cells_history)

            # Plot the lineage tree
            plot_hierarchical_tree_with_progress(lineage, initial_cell_id, time_steps, canvas, figure)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for the cell ID.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Bind the button to the plot_tree function
    plot_button.config(command=plot_tree)

    # Plot the default tree
    plot_tree()

    # Run the GUI event loop
    root.mainloop()

# Run the GUI
run_gui()