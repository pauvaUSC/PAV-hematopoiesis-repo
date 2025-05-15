import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from hematopoiesis_model_v3 import simulate_hematopoiesis, params

# Global variable to store the current figure
current_figure = None

def run_simulation():
    global current_figure
    try:
        # Disable the Run button and show a "Running" message
        run_button.config(state="disabled")
        status_label.config(text="Simulation Running...", foreground="blue")
        root.update_idletasks()  # Update the GUI to reflect changes

        # Get user inputs for initial conditions
        initial_lt_hsc_count = int(lt_hsc_count_entry.get())
        initial_lt_quiescent = int(lt_quiescent_entry.get())
        initial_st_hsc_count = int(st_hsc_count_entry.get())
        initial_st_quiescent = int(st_quiescent_entry.get())
        time_steps = int(time_steps_entry.get())
        label_fraction = float(label_fraction_entry.get())
        seed = int(seed_entry.get())

        # Update model parameters from user inputs
        for param, entry in param_entries.items():
            params[param] = float(entry.get())

        # Generate sample days
        sample_days = list(range(0, time_steps, 7))

        # Run the simulation
        counts, cells_history, division_events, apoptosis_events = simulate_hematopoiesis(
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

        # Plot results in the GUI
        current_figure = plot_results(sample_days, counts)

        # Update the status to indicate completion
        status_label.config(text="Simulation Completed!", foreground="green")

    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"An error occurred: {e}")
        status_label.config(text="Simulation Failed!", foreground="red")

    finally:
        # Re-enable the Run button
        run_button.config(state="normal")

def plot_results(sample_days, counts):
    # Clear only the plot area (not the "Save Plot" button)
    for widget in plot_frame.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg) or isinstance(widget, ttk.Label):  # Remove plot canvas and placeholder
            widget.destroy()

    # LT-HSC counts
    time_points = sample_days
    lt_hsc_active = [labeled + unlabeled for labeled, unlabeled in zip(counts["LT-HSC_active_labeled"], counts["LT-HSC_active_unlabeled"])]
    lt_hsc_quiescent = [labeled + unlabeled for labeled, unlabeled in zip(counts["LT-HSC_quiescent_labeled"], counts["LT-HSC_quiescent_unlabeled"])]
    lt_total = [active + quiescent for active, quiescent in zip(lt_hsc_active, lt_hsc_quiescent)]

    # ST-HSC counts
    st_hsc_active = [labeled + unlabeled for labeled, unlabeled in zip(counts["ST-HSC_active_labeled"], counts["ST-HSC_active_unlabeled"])]
    st_hsc_quiescent = [labeled + unlabeled for labeled, unlabeled in zip(counts["ST-HSC_quiescent_labeled"], counts["ST-HSC_quiescent_unlabeled"])]
    st_total = [active + quiescent for active, quiescent in zip(st_hsc_active, st_hsc_quiescent)]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Plot LT-HSC counts
    axs[0].plot(time_points, lt_hsc_active, label="Active LT-HSCs", marker="o")
    axs[0].plot(time_points, lt_hsc_quiescent, label="Quiescent LT-HSCs", marker="s")
    axs[0].plot(time_points, lt_total, label="Total LT-HSCs", marker="d")
    axs[0].set_xlabel("Time (days)")
    axs[0].set_ylabel("Number of LT-HSCs")
    axs[0].set_title("LT-HSC Counts Over Time")
    axs[0].legend()
    axs[0].grid(True)

    # Plot ST-HSC counts
    axs[1].plot(time_points, st_hsc_active, label="Active ST-HSCs", marker="o")
    axs[1].plot(time_points, st_hsc_quiescent, label="Quiescent ST-HSCs", marker="s")
    axs[1].plot(time_points, st_total, label="Total ST-HSCs", marker="d")
    axs[1].set_xlabel("Time (days)")
    axs[1].set_ylabel("Number of ST-HSCs")
    axs[1].set_title("ST-HSC Counts Over Time")
    axs[1].legend()
    axs[1].grid(True)

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    return fig  # Return the figure for saving

def save_plot():
    global current_figure
    if current_figure is None:
        messagebox.showwarning("No Plot", "No plot available to save. Please run the simulation first.")
        return

    # Ask the user where to save the plot
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"),
                                                        ("JPEG files", "*.jpg"),
                                                        ("All files", "*.*")])
    if file_path:
        current_figure.savefig(file_path)
        messagebox.showinfo("Plot Saved", f"Plot saved successfully to {file_path}")

# Create the main window
root = tk.Tk()
root.title("Hematopoiesis Model Simulation")

# Create input fields for initial conditions
input_frame = ttk.Frame(root)
input_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

ttk.Label(input_frame, text="Initial LT-HSC Active:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
lt_hsc_count_entry = ttk.Entry(input_frame)
lt_hsc_count_entry.grid(row=0, column=1, padx=10, pady=5)
lt_hsc_count_entry.insert(0, "500")

ttk.Label(input_frame, text="Initial LT-HSC Quiescent:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
lt_quiescent_entry = ttk.Entry(input_frame)
lt_quiescent_entry.grid(row=1, column=1, padx=10, pady=5)
lt_quiescent_entry.insert(0, "1500")

ttk.Label(input_frame, text="Initial ST-HSC Active:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
st_hsc_count_entry = ttk.Entry(input_frame)
st_hsc_count_entry.grid(row=2, column=1, padx=10, pady=5)
st_hsc_count_entry.insert(0, "5000")

ttk.Label(input_frame, text="Initial ST-HSC Quiescent:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
st_quiescent_entry = ttk.Entry(input_frame)
st_quiescent_entry.grid(row=3, column=1, padx=10, pady=5)
st_quiescent_entry.insert(0, "5000")

ttk.Label(input_frame, text="Time Steps:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
time_steps_entry = ttk.Entry(input_frame)
time_steps_entry.grid(row=4, column=1, padx=10, pady=5)
time_steps_entry.insert(0, "50")

ttk.Label(input_frame, text="Fraction of LT-HSCs that are initially labeled:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
label_fraction_entry = ttk.Entry(input_frame)
label_fraction_entry.grid(row=5, column=1, padx=10, pady=5)
label_fraction_entry.insert(0, "0.001")

ttk.Label(input_frame, text="Seed:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
seed_entry = ttk.Entry(input_frame)
seed_entry.grid(row=6, column=1, padx=10, pady=5)
seed_entry.insert(0, "42")


# Add a section for model parameters
param_frame = ttk.LabelFrame(input_frame, text="Model Parameters")
param_frame.grid(row=7, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

# Subdivide parameters into sections
sections = {
    "Cell Cycle Parameters": [
        "LT_HSC_DIVISION_THRESHOLD",
        "ST_HSC_DIVISION_THRESHOLD",
        "LT_HSC_TIME_MEAN",
        "LT_HSC_TIME_STD",
        "LT_HSC_DAILY_DIVISION_RATE",
        "ST_HSC_TIME_MEAN",
        "ST_HSC_TIME_STD",
        "ST_HSC_DAILY_DIVISION_RATE"
    ],
    "Division Probabilities": [
        "P1A", "P2A", "P3A", "P4A",
        "P1B", "P2B", "P3B", "P4B"
    ],
    "Labeling Parameters": [
        "LABEL_THRESHOLD",
        "LABELING_PERIOD",
        "LABELING_PROB",
    ],
    "Quiescence and Apoptosis Parameters": [
        "PQA", "PQB",
        "P_EXIT_QUIESCENCE_A",
        "P_EXIT_QUIESCENCE_B",
        "PAQA", "PAQB",
        "PAA", "PAB"
    ]
}

# Mapping of parameter names to descriptive labels
param_labels = {
    "LT_HSC_DIVISION_THRESHOLD": "LT-HSC Division Threshold",
    "ST_HSC_DIVISION_THRESHOLD": "ST-HSC Division Threshold",
    "PAA": "Probability of Apoptosis for Active LT-HSCs",
    "PAB": "Probability of Apoptosis for Active ST-HSCs",
    "PAQA": "Probability of Apoptosis for Quiescent LT-HSCs",
    "PAQB": "Probability of Apoptosis for Quiescent ST-HSCs",
    "PQA": "Probability of LT-HSCs Entering Quiescence",
    "PQB": "Probability of ST-HSCs Entering Quiescence",
    "P_EXIT_QUIESCENCE_A": "Probability of LT-HSCs Exiting Quiescence",
    "P_EXIT_QUIESCENCE_B": "Probability of ST-HSCs Exiting Quiescence",
    "P1A": "Probability of Symmetric Self-Renewal for LT-HSCs",
    "P2A": "Probability of Asymmetric Division for LT-HSCs",
    "P3A": "Probability of Symmetric Differentiation for LT-HSCs",
    "P4A": "Probability of Full Differentiation for LT-HSCs",
    "P1B": "Probability of Symmetric Self-Renewal for ST-HSCs",
    "P2B": "Probability of Asymmetric Division for ST-HSCs",
    "P3B": "Probability of Symmetric Differentiation for ST-HSCs",
    "P4B": "Probability of Full Differentiation for ST-HSCs",
    "LT_HSC_TIME_MEAN": "Mean Division Time for LT-HSCs (days)",
    "LT_HSC_TIME_STD": "Standard Deviation of Division Time for LT-HSCs (days)",
    "LT_HSC_DAILY_DIVISION_RATE": "Daily Division Rate for LT-HSCs",
    "ST_HSC_TIME_MEAN": "Mean Division Time for ST-HSCs (days)",
    "ST_HSC_TIME_STD": "Standard Deviation of Division Time for ST-HSCs (days)",
    "ST_HSC_DAILY_DIVISION_RATE": "Daily Division Rate for ST-HSCs",
    "LABEL_THRESHOLD": "Labeling Threshold",
    "LABELING_PERIOD": "Labeling Period (days)",
    "LABELING_PROB": "Probability of Labeling During Labeling Period"
}

# Define a custom style for bold section names
style = ttk.Style()
style.configure("Bold.TLabelframe.Label", font=("TkDefaultFont", 12, "bold"))

# Add parameters to the GUI, grouped by sections
param_entries = {}
for section, parameters in sections.items():
    # Apply the bold style to the section label
    section_frame = ttk.LabelFrame(param_frame, text=section, style="Bold.TLabelframe")
    section_frame.pack(fill=tk.X, padx=5, pady=5)

    for param in parameters:
        ttk.Label(section_frame, text=f"{param_labels[param]}:").grid(row=parameters.index(param), column=0, padx=5, pady=2, sticky="w")
        entry = ttk.Entry(section_frame)
        entry.grid(row=parameters.index(param), column=1, padx=5, pady=2)
        entry.insert(0, str(params[param]))
        param_entries[param] = entry

# Add a button to run the simulation
run_button = ttk.Button(input_frame, text="Run Simulation", command=run_simulation)
run_button.grid(row=8, column=0, columnspan=2, pady=10)

# Create a frame for the plot with a placeholder
plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Add a "Save Plot" button to the top of the plot area
save_button = ttk.Button(plot_frame, text="Save Plot", command=save_plot)
save_button.pack(side=tk.TOP, padx=10, pady=5)

# Add a placeholder label for the plot area
placeholder_label = ttk.Label(plot_frame, text="Plots will appear here", foreground="gray", anchor="center")
placeholder_label.pack(fill=tk.BOTH, expand=True)

# Add a status label to show the simulation status
status_label = ttk.Label(input_frame, text="", foreground="blue")
status_label.grid(row=10, column=0, columnspan=2, pady=5)

# Create a frame for the plot with a placeholder
# plot_frame = ttk.Frame(root)
# plot_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
# placeholder_label = ttk.Label(plot_frame, text="Plots will appear here", foreground="gray", anchor="center")
# placeholder_label.pack(fill=tk.BOTH, expand=True)

# Start the main loop
root.mainloop()