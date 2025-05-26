import pickle
import numpy as np
import matplotlib.pyplot as plt

max_weeks = 100


filenames = [
    ("hematopoiesis_simulation_data_normal.pkl", "Normal", "#87ceeb", "#1f77b4", "-"),   # light blue, dark blue
    ("hematopoiesis_simulation_data_stress1.pkl", "Stress 1", "#90ee90", "#228B22", "-"), # light green, dark green
    ("hematopoiesis_simulation_data_stress2.pkl", "Stress 2", "#ffb6b6", "#d62728", "--"), # light red, dark red (dashed)
]

#### ST- active ####
key_labeled = "ST-HSC_active_labeled"
key_unlabeled = "ST-HSC_active_unlabeled"

plt.figure(figsize=(10, 3))

for fname, label, light_color, dark_color, linestyle in filenames:
    with open(fname, "rb") as f:
        data = pickle.load(f)
    all_counts = data["counts"]

    all_traces = []
    for seed_counts in all_counts:
        if key_labeled in seed_counts and key_unlabeled in seed_counts:
            total = np.array(seed_counts[key_labeled]) + np.array(seed_counts[key_unlabeled])
            total = total[:max_weeks]  # Trim to max_weeks
            all_traces.append(total)
    all_traces = np.array(all_traces)

    # Plot each seed in light color
    for trace in all_traces:
        plt.plot(trace, color=light_color, alpha=0.2, linewidth=1, linestyle=linestyle)

    # Plot mean in dark color
    mean_trace = np.mean(all_traces, axis=0)
    plt.plot(mean_trace, color=dark_color, alpha=1, linewidth=2, linestyle=linestyle, label=label)

plt.xlabel("Time (weeks)")
plt.ylabel("Cell count")
plt.title("ST-HSC Active")
plt.legend()
plt.tight_layout()
plt.savefig("st_hsc_active.png", dpi=300)  # Save the figure as a PNG file
plt.show()


#### LT- active ####
key_labeled = "LT-HSC_active_labeled"
key_unlabeled = "LT-HSC_active_unlabeled"

plt.figure(figsize=(10, 3))

for fname, label, light_color, dark_color, linestyle in filenames:
    with open(fname, "rb") as f:
        data = pickle.load(f)
    all_counts = data["counts"]

    all_traces = []
    for seed_counts in all_counts:
        if key_labeled in seed_counts and key_unlabeled in seed_counts:
            total = np.array(seed_counts[key_labeled]) + np.array(seed_counts[key_unlabeled])
            total = total[:max_weeks]  # Trim to max_weeks
            all_traces.append(total)
    all_traces = np.array(all_traces)

    # Plot each seed in light color
    for trace in all_traces:
        plt.plot(trace, color=light_color, alpha=0.2, linewidth=1, linestyle=linestyle)

    # Plot mean in dark color
    mean_trace = np.mean(all_traces, axis=0)
    plt.plot(mean_trace, color=dark_color, alpha=1, linewidth=2, linestyle=linestyle, label=label)


plt.xlabel("Time (weeks)")
plt.ylabel("Cell count")
plt.title("LT-HSC Active")
plt.legend()
plt.tight_layout()
plt.savefig("lt_hsc_active.png", dpi=300)  # Save the figure as a PNG file
plt.show()


#### ST- quiescent ####
key_labeled = "ST-HSC_quiescent_labeled"
key_unlabeled = "ST-HSC_quiescent_unlabeled"

plt.figure(figsize=(10, 3))

for fname, label, light_color, dark_color, linestyle in filenames:
    with open(fname, "rb") as f:
        data = pickle.load(f)
    all_counts = data["counts"]

    all_traces = []
    for seed_counts in all_counts:
        if key_labeled in seed_counts and key_unlabeled in seed_counts:
            total = np.array(seed_counts[key_labeled]) + np.array(seed_counts[key_unlabeled])
            total = total[:max_weeks]  # Trim to max_weeks
            all_traces.append(total)
    all_traces = np.array(all_traces)

    # Plot each seed in light color
    for trace in all_traces:
        plt.plot(trace, color=light_color, alpha=0.2, linewidth=1, linestyle=linestyle)

    # Plot mean in dark color
    mean_trace = np.mean(all_traces, axis=0)
    plt.plot(mean_trace, color=dark_color, alpha=1, linewidth=2, linestyle=linestyle, label=label)

plt.xlabel("Time (weeks)")
plt.ylabel("Cell count")
plt.title("ST-HSC Quiescent")
plt.legend()
plt.tight_layout()
plt.savefig("st_hsc_quiescent.png", dpi=300)  # Save the figure as a PNG file
plt.show()


#### ST- quiescent ####
key_labeled = "LT-HSC_quiescent_labeled"
key_unlabeled = "LT-HSC_quiescent_unlabeled"

plt.figure(figsize=(10, 3))

for fname, label, light_color, dark_color, linestyle in filenames:
    with open(fname, "rb") as f:
        data = pickle.load(f)
    all_counts = data["counts"]

    all_traces = []
    for seed_counts in all_counts:
        if key_labeled in seed_counts and key_unlabeled in seed_counts:
            total = np.array(seed_counts[key_labeled]) + np.array(seed_counts[key_unlabeled])
            total = total[:max_weeks]  # Trim to max_weeks
            all_traces.append(total)
    all_traces = np.array(all_traces)

    # Plot each seed in light color
    for trace in all_traces:
        plt.plot(trace, color=light_color, alpha=0.2, linewidth=1, linestyle=linestyle)

    # Plot mean in dark color
    mean_trace = np.mean(all_traces, axis=0)
    plt.plot(mean_trace, color=dark_color, alpha=1, linewidth=2, linestyle=linestyle, label=label)

plt.xlabel("Time (weeks)")
plt.ylabel("Cell count")
plt.title("LT-HSC Quiescent")
plt.legend()
plt.tight_layout()
plt.savefig("lt_hsc_quiescent.png", dpi=300)  # Save the figure as a PNG file
plt.show()