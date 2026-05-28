"""
plot_lineage_tree_gui_single_png.py

Single-purpose GUI for plotting one time-scaled hematopoiesis lineage tree
and saving exactly one PNG file for the requested root cell.

What this script does:
  - Loads simulation_data.pkl.
  - Lets the user type one root cell ID.
  - Traces that cell and all descendants.
  - Plots the time-scaled lineage tree in the GUI.
  - Saves one PNG named lineage_tree_cell_<root_id>.png.

What this script does not do:
  - No HTML output.
  - No PDF output.
  - No batch loop over multiple roots.
  - No separate legend figure.
  - No composite figure.

Recommended folder layout:
  your_folder/
      plot_lineage_tree_gui_single_png.py
      simulation_data.pkl
      hematopoiesis_model_v5.py   # needed if the pickle contains Cell objects

Dependencies:
  pip install matplotlib

Tkinter is included with many Python installs. On some Linux systems it may need
installation through the system package manager, for example: python3-tk.
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make sure the script folder is importable so pickle can find
# hematopoiesis_model_v5.Cell when simulation_data.pkl contains Cell objects.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import hematopoiesis_model_v5 as model_v5  # noqa: F401
except Exception:
    model_v5 = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure


# -----------------------------------------------------------------------------
# User-editable defaults
# -----------------------------------------------------------------------------

DEFAULT_PKL_FILE = "simulation_data.pkl"
DEFAULT_ROOT_CELL_ID = 47
DEFAULT_TIME_STEPS = 7 * 50  # days; the GUI will try to infer this from counts
DEFAULT_TIME_UNIT = "weeks"  # "weeks" or "days"
DEFAULT_MAX_CELLS_TO_PLOT = 2500  # blank in the GUI means plot the full clone

PNG_DPI = 300
FIGURE_WIDTH_IN = 12.5
FIGURE_MIN_HEIGHT_IN = 4.2
FIGURE_MAX_HEIGHT_IN = 11.0
FIGURE_ROW_HEIGHT_IN = 0.24

LINE_WIDTH = 2.6
CONNECTOR_WIDTH = 1.2
TERMINAL_SIZE = 58
ROOT_TERMINAL_SIZE = 82
BIRTH_MARKER_SIZE = 42
ROW_SPACING = 1.0

TYPE_COLOR = {
    "LT-HSC": "#2563EB",  # blue
    "ST-HSC": "#16A34A",  # green
    "MPP": "#DC2626",     # red
}

TYPE_LIGHT_COLOR = {
    "LT-HSC": "#93C5FD",
    "ST-HSC": "#86EFAC",
    "MPP": "#FCA5A5",
}

STATE_LINESTYLE = {
    "Non-quiescent Active": "solid",
    "Quiescent": (0, (1, 2)),
    "Non-quiescent Inactive": (0, (4, 2)),
    "Removed": "solid",
}

TERMINAL_MARKER = {
    "Non-quiescent Active": "o",
    "Quiescent": "D",
    "Non-quiescent Inactive": "o",
    "Removed": "x",
}

STATE_LABEL = {
    "Non-quiescent Active": "alive active",
    "Quiescent": "alive quiescent",
    "Non-quiescent Inactive": "alive inactive",
    "Removed": "removed",
}


# -----------------------------------------------------------------------------
# Pickle loading and data normalization
# -----------------------------------------------------------------------------

class CompatibleUnpickler(pickle.Unpickler):
    """Help load older pickles made when Cell lived in __main__."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "Cell" and model_v5 is not None:
            if hasattr(model_v5, "Cell"):
                return model_v5.Cell
        return super().find_class(module, name)


def _looks_like_active_cells(x: Any) -> bool:
    return isinstance(x, list) and (not x or hasattr(x[0], "unique_id"))


def _looks_like_history(x: Any) -> bool:
    return isinstance(x, list) and (not x or isinstance(x[0], dict) and "unique_id" in x[0])


def load_data(pkl_file: str | os.PathLike[str]) -> tuple[Any, list[Any], list[dict[str, Any]], Any, Any]:
    """
    Load simulation_data.pkl robustly.

    Accepted tuple orders:
      1. (counts, cells, cells_history, division_events, apoptosis_events)
      2. (counts, cells_history, division_events, apoptosis_events, cells)
    """
    pkl_path = Path(pkl_file)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Could not find '{pkl_path}'.")

    # Add the pickle folder to sys.path as well, in case hematopoiesis_model_v5.py
    # is next to the data file rather than next to this GUI script.
    data_dir = str(pkl_path.resolve().parent)
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)

    global model_v5
    if model_v5 is None:
        try:
            import hematopoiesis_model_v5 as imported_model_v5  # noqa: F401
            model_v5 = imported_model_v5
        except Exception:
            model_v5 = None

    with pkl_path.open("rb") as f:
        data = CompatibleUnpickler(f).load()

    if not isinstance(data, tuple) or len(data) != 5:
        raise ValueError(
            "Expected a 5-item pickle tuple: either "
            "(counts, cells, cells_history, division_events, apoptosis_events) "
            "or (counts, cells_history, division_events, apoptosis_events, cells)."
        )

    counts = data[0]

    if _looks_like_active_cells(data[1]) and _looks_like_history(data[2]):
        cells = data[1]
        cells_history = data[2]
        division_events = data[3]
        apoptosis_events = data[4]
    elif _looks_like_history(data[1]) and _looks_like_active_cells(data[4]):
        cells_history = data[1]
        division_events = data[2]
        apoptosis_events = data[3]
        cells = data[4]
    else:
        raise ValueError(
            "Could not determine the pickle tuple order. "
            "Please inspect how simulation_data.pkl was saved."
        )

    return counts, cells, cells_history, division_events, apoptosis_events


def _active_cell_to_record(cell: Any) -> dict[str, Any]:
    return {
        "unique_id": int(cell.unique_id),
        "parent_id": cell.parent_id,
        "cell_type": getattr(cell, "cell_type", "Unknown"),
        "state": getattr(cell, "state", "Unknown"),
        "division_count": getattr(cell, "division_count", None),
        "creation_time": getattr(cell, "creation_time", 0),
        "end_time": None,
        "end_reason": getattr(cell, "end_reason", None),
        "label_intensity": getattr(cell, "label_intensity", None),
    }


def _history_cell_to_record(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "unique_id": int(c["unique_id"]),
        "parent_id": c.get("parent_id"),
        "cell_type": c.get("cell_type", "Unknown"),
        "state": "Removed",
        "division_count": c.get("division_count"),
        "creation_time": c.get("start_time", c.get("creation_time", 0)),
        "end_time": c.get("end_time"),
        "end_reason": c.get("end_reason"),
        "label_intensity": c.get("label_intensity"),
    }


def collect_all_cells(cells: list[Any], cells_history: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Merge active cells and removed-cell history into one metadata dictionary."""
    all_cells: dict[int, dict[str, Any]] = {}

    for cell in cells:
        record = _active_cell_to_record(cell)
        all_cells[record["unique_id"]] = record

    for history_cell in cells_history:
        record = _history_cell_to_record(history_cell)
        all_cells[record["unique_id"]] = record

    return all_cells


def infer_time_steps(counts: Any, all_cells: dict[int, dict[str, Any]], fallback: int = DEFAULT_TIME_STEPS) -> int:
    """
    Infer simulation length in days.

    The model typically samples weekly with sample_days = range(0, time_steps, 7),
    so len(count_series) * 7 is a good estimate. We also make sure the inferred
    value is not earlier than any observed creation/end time.
    """
    inferred: int | None = None

    if isinstance(counts, dict):
        lengths = [len(v) for v in counts.values() if hasattr(v, "__len__") and not isinstance(v, str)]
        if lengths and max(lengths) > 0:
            inferred = max(lengths) * 7

    if inferred is None:
        inferred = fallback

    observed_times: list[float] = []
    for cell in all_cells.values():
        for key in ("creation_time", "end_time"):
            value = cell.get(key)
            if value is not None:
                try:
                    observed_times.append(float(value))
                except (TypeError, ValueError):
                    pass

    if observed_times:
        inferred = max(inferred, int(max(observed_times)))

    return int(inferred)


# -----------------------------------------------------------------------------
# Lineage tracing
# -----------------------------------------------------------------------------

def build_children_map(all_cells: dict[int, dict[str, Any]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = defaultdict(list)

    for uid, meta in all_cells.items():
        parent_id = meta.get("parent_id")
        if parent_id is not None and parent_id in all_cells:
            children[int(parent_id)].append(int(uid))

    for parent_id in children:
        children[parent_id].sort(
            key=lambda child_id: (
                all_cells[child_id].get("creation_time") is None,
                all_cells[child_id].get("creation_time") or 0,
                all_cells[child_id].get("cell_type") or "",
                child_id,
            )
        )

    return children


def trace_lineage(root_id: int, all_cells: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the root cell plus all descendants in root-first DFS order."""
    if root_id not in all_cells:
        raise ValueError(f"Root cell {root_id} was not found in active cells or history.")

    children = build_children_map(all_cells)
    lineage: list[dict[str, Any]] = []
    visited: set[int] = set()

    def dfs(uid: int) -> None:
        if uid in visited:
            return
        visited.add(uid)
        lineage.append(all_cells[uid])
        for child_uid in children.get(uid, []):
            dfs(child_uid)

    dfs(root_id)
    return lineage


def limit_lineage(lineage: list[dict[str, Any]], max_cells: int | None) -> tuple[list[dict[str, Any]], bool]:
    if max_cells is None or max_cells <= 0 or len(lineage) <= max_cells:
        return lineage, False
    return lineage[:max_cells], True


def lineage_summary(lineage: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    n_nodes = len(lineage)
    n_removed = sum(1 for cell in lineage if cell.get("state") == "Removed")
    n_alive = n_nodes - n_removed
    n_mpp = sum(1 for cell in lineage if cell.get("cell_type") == "MPP")
    return n_nodes, n_alive, n_removed, n_mpp


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def time_scale(value: float | int | None, time_unit: str) -> float | None:
    if value is None:
        return None
    if time_unit.lower().startswith("week"):
        return float(value) / 7.0
    return float(value)


def axis_title(time_unit: str) -> str:
    if time_unit.lower().startswith("week"):
        return "Time after simulation start (weeks)"
    return "Time after simulation start (days)"


def height_for_lineage(lineage: list[dict[str, Any]]) -> float:
    return max(
        FIGURE_MIN_HEIGHT_IN,
        min(FIGURE_MAX_HEIGHT_IN, FIGURE_ROW_HEIGHT_IN * len(lineage) + 1.1),
    )


def cell_end_time(cell: dict[str, Any], time_steps: int) -> float:
    end_time = cell.get("end_time")
    if end_time is None:
        end_time = time_steps
    return float(end_time)


def assign_y_positions(lineage: list[dict[str, Any]]) -> dict[int, float]:
    return {int(cell["unique_id"]): -i * ROW_SPACING for i, cell in enumerate(lineage)}


def draw_lineage_tree(
    figure: Figure,
    lineage: list[dict[str, Any]],
    root_id: int,
    time_steps: int,
    time_unit: str = DEFAULT_TIME_UNIT,
) -> None:
    """Draw the lineage tree on a Matplotlib Figure."""
    if not lineage:
        raise ValueError("Lineage is empty.")

    figure.clear()
    figure.set_size_inches(FIGURE_WIDTH_IN, height_for_lineage(lineage), forward=True)
    ax = figure.add_subplot(111)

    lineage_ids = {int(cell["unique_id"]) for cell in lineage}
    y_pos = assign_y_positions(lineage)

    horizontal_segments: dict[tuple[str, str], list[list[tuple[float, float]]]] = defaultdict(list)
    connector_segments: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    birth_points: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    terminal_points: dict[tuple[str, str], tuple[list[float], list[float], list[float]]] = defaultdict(lambda: ([], [], []))

    for cell in lineage:
        uid = int(cell["unique_id"])
        cell_type = cell.get("cell_type", "Unknown")
        state = cell.get("state", "Unknown")
        born = float(cell.get("creation_time") or 0)
        ended = cell_end_time(cell, time_steps)
        y = y_pos[uid]

        x0 = time_scale(born, time_unit)
        x1 = time_scale(ended, time_unit)
        if x0 is None or x1 is None:
            continue

        horizontal_segments[(cell_type, state)].append([(x0, y), (x1, y)])

        xs, ys, sizes = terminal_points[(cell_type, state)]
        xs.append(x1)
        ys.append(y)
        sizes.append(ROOT_TERMINAL_SIZE if uid == root_id else TERMINAL_SIZE)

    for child in lineage:
        child_uid = int(child["unique_id"])
        parent_uid = child.get("parent_id")
        if parent_uid is None or parent_uid not in lineage_ids:
            continue

        birth = float(child.get("creation_time") or 0)
        x = time_scale(birth, time_unit)
        if x is None:
            continue

        child_type = child.get("cell_type", "Unknown")
        connector_segments[child_type].append([(x, y_pos[int(parent_uid)]), (x, y_pos[child_uid])])

        xs, ys = birth_points[child_type]
        xs.append(x)
        ys.append(y_pos[child_uid])

    # Birth connectors first, behind the lifetimes.
    for cell_type, segments in connector_segments.items():
        color = TYPE_COLOR.get(cell_type, "#6B7280")
        collection = LineCollection(segments, colors=color, linewidths=CONNECTOR_WIDTH, alpha=0.55, zorder=1)
        ax.add_collection(collection)

    # Lifetime bars.
    ordered_segment_keys = sorted(
        horizontal_segments.keys(),
        key=lambda key: (
            list(TYPE_COLOR).index(key[0]) if key[0] in TYPE_COLOR else 99,
            key[1],
        ),
    )
    for cell_type, state in ordered_segment_keys:
        color = TYPE_COLOR.get(cell_type, "#6B7280")
        linestyle = STATE_LINESTYLE.get(state, "solid")
        collection = LineCollection(
            horizontal_segments[(cell_type, state)],
            colors=color,
            linewidths=LINE_WIDTH,
            linestyles=linestyle,
            zorder=2,
        )
        ax.add_collection(collection)

    # Birth markers.
    for cell_type, (xs, ys) in birth_points.items():
        ax.scatter(
            xs,
            ys,
            s=BIRTH_MARKER_SIZE,
            marker="o",
            facecolors=TYPE_LIGHT_COLOR.get(cell_type, "#D1D5DB"),
            edgecolors=TYPE_COLOR.get(cell_type, "#6B7280"),
            linewidths=0.8,
            zorder=3,
        )

    # Terminal markers.
    ordered_terminal_keys = sorted(
        terminal_points.keys(),
        key=lambda key: (
            list(TYPE_COLOR).index(key[0]) if key[0] in TYPE_COLOR else 99,
            key[1],
        ),
    )
    for cell_type, state in ordered_terminal_keys:
        xs, ys, sizes = terminal_points[(cell_type, state)]
        color = TYPE_COLOR.get(cell_type, "#6B7280")
        light_color = TYPE_LIGHT_COLOR.get(cell_type, "#D1D5DB")
        marker = TERMINAL_MARKER.get(state, "o")

        if state == "Removed":
            ax.scatter(xs, ys, s=sizes, marker=marker, color=color, linewidths=1.3, zorder=4)
        elif state == "Non-quiescent Inactive":
            ax.scatter(xs, ys, s=sizes, marker=marker, facecolors="white", edgecolors=color, linewidths=1.3, zorder=4)
        else:
            ax.scatter(xs, ys, s=sizes, marker=marker, facecolors=light_color, edgecolors=color, linewidths=1.3, zorder=4)

    x_end = time_scale(time_steps, time_unit)
    if x_end is None:
        x_end = float(time_steps)

    min_y = min(y_pos.values())
    max_y = max(y_pos.values())
    ax.axvline(x_end, color="#94A3B8", linewidth=1.0, linestyle=(0, (1, 2)), zorder=0)
    ax.text(x_end, max_y + 0.65, f"{time_steps} days", ha="center", va="bottom", fontsize=10, color="#64748B")

    x_padding = max(1.0, x_end * 0.02)
    ax.set_xlim(left=0, right=x_end + x_padding)
    ax.set_ylim(bottom=min_y - 1.0, top=max_y + 1.15)
    ax.set_xlabel(axis_title(time_unit), fontsize=10)
    ax.set_yticks([])
    ax.set_ylabel("")

    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    figure.tight_layout()



# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

class LineageTreeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Single Lineage Tree PNG Exporter")
        self.root.geometry("1180x780")

        self.loaded_pkl_path: Path | None = None
        self.counts: Any = None
        self.all_cells: dict[int, dict[str, Any]] | None = None
        self.inferred_time_steps = DEFAULT_TIME_STEPS

        default_pkl_path = SCRIPT_DIR / DEFAULT_PKL_FILE
        if not default_pkl_path.exists():
            default_pkl_path = Path(DEFAULT_PKL_FILE)

        self.pkl_path_var = tk.StringVar(value=str(default_pkl_path))
        self.output_dir_var = tk.StringVar(value="")
        self.root_id_var = tk.StringVar(value=str(DEFAULT_ROOT_CELL_ID))
        self.time_steps_var = tk.StringVar(value=str(DEFAULT_TIME_STEPS))
        self.max_cells_var = tk.StringVar(value=str(DEFAULT_MAX_CELLS_TO_PLOT))
        self.status_var = tk.StringVar(value="Choose a root cell ID, then click Plot and Save PNG.")

        self._build_controls()
        self._build_plot_area()

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.root, padding=(10, 8))
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="Data file:").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        data_entry = ttk.Entry(controls, textvariable=self.pkl_path_var, width=58)
        data_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(controls, text="Browse", command=self.browse_pkl).grid(row=0, column=2, padx=(0, 12), pady=2)

        ttk.Label(controls, text="Root cell ID:").grid(row=0, column=3, sticky="w", padx=(0, 5), pady=2)
        root_entry = ttk.Entry(controls, textvariable=self.root_id_var, width=12)
        root_entry.grid(row=0, column=4, sticky="w", padx=(0, 12), pady=2)
        root_entry.bind("<Return>", lambda _event: self.plot_and_save())

        ttk.Button(controls, text="Plot and Save PNG", command=self.plot_and_save).grid(row=0, column=5, padx=(0, 5), pady=2)

        ttk.Label(controls, text="Output folder:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=2)
        output_entry = ttk.Entry(controls, textvariable=self.output_dir_var, width=58)
        output_entry.grid(row=1, column=1, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(controls, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=(0, 12), pady=2)

        ttk.Label(controls, text="Simulation days:").grid(row=1, column=3, sticky="w", padx=(0, 5), pady=2)
        ttk.Entry(controls, textvariable=self.time_steps_var, width=12).grid(row=1, column=4, sticky="w", padx=(0, 12), pady=2)

        ttk.Label(controls, text="Max cells:").grid(row=1, column=5, sticky="e", padx=(0, 5), pady=2)
        ttk.Entry(controls, textvariable=self.max_cells_var, width=10).grid(row=1, column=6, sticky="w", pady=2)

        status = ttk.Label(controls, textvariable=self.status_var, anchor="w")
        status.grid(row=2, column=0, columnspan=7, sticky="ew", pady=(6, 0))

        controls.columnconfigure(1, weight=1)

    def _build_plot_area(self) -> None:
        plot_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(FIGURE_WIDTH_IN, FIGURE_MIN_HEIGHT_IN), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        self._draw_welcome_message()

    def _draw_welcome_message(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "Enter a root cell ID and click Plot and Save PNG",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.text(
            0.5,
            0.45,
            "The output will be one PNG file: lineage_tree_cell_<root_id>.png",
            ha="center",
            va="center",
            fontsize=10,
            color="#64748B",
        )
        self.canvas.draw_idle()

    def browse_pkl(self) -> None:
        initial_dir = Path(self.pkl_path_var.get()).expanduser().parent
        filename = filedialog.askopenfilename(
            title="Choose simulation_data.pkl",
            initialdir=str(initial_dir) if initial_dir.exists() else str(SCRIPT_DIR),
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
        )
        if filename:
            self.pkl_path_var.set(filename)
            self.loaded_pkl_path = None

    def browse_output_dir(self) -> None:
        current = self.output_dir_var.get().strip()
        initial_dir = Path(current).expanduser() if current else Path(self.pkl_path_var.get()).expanduser().parent
        dirname = filedialog.askdirectory(
            title="Choose output folder",
            initialdir=str(initial_dir) if initial_dir.exists() else str(SCRIPT_DIR),
        )
        if dirname:
            self.output_dir_var.set(dirname)

    def ensure_data_loaded(self) -> None:
        pkl_path = Path(self.pkl_path_var.get()).expanduser()
        if not pkl_path.is_absolute():
            pkl_path = (Path.cwd() / pkl_path).resolve()

        if self.loaded_pkl_path == pkl_path and self.all_cells is not None:
            return

        self.status_var.set(f"Loading {pkl_path.name}...")
        self.root.update_idletasks()

        counts, cells, cells_history, _division_events, _apoptosis_events = load_data(pkl_path)
        self.counts = counts
        self.all_cells = collect_all_cells(cells, cells_history)
        self.loaded_pkl_path = pkl_path
        self.inferred_time_steps = infer_time_steps(counts, self.all_cells, DEFAULT_TIME_STEPS)
        self.time_steps_var.set(str(self.inferred_time_steps))

        self.status_var.set(
            f"Loaded {len(cells):,} active cells and {len(cells_history):,} removed/history cells. "
            f"Inferred simulation length: {self.inferred_time_steps} days."
        )
        self.root.update_idletasks()

    def parse_root_id(self) -> int:
        try:
            return int(self.root_id_var.get().strip())
        except ValueError as exc:
            raise ValueError("Please enter a valid integer root cell ID.") from exc

    def parse_time_steps(self) -> int:
        try:
            value = int(self.time_steps_var.get().strip())
        except ValueError as exc:
            raise ValueError("Please enter simulation days as an integer.") from exc
        if value <= 0:
            raise ValueError("Simulation days must be greater than 0.")
        return value

    def parse_max_cells(self) -> int | None:
        raw = self.max_cells_var.get().strip()
        if raw == "":
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("Max cells must be an integer, or blank to plot all cells.") from exc
        if value <= 0:
            return None
        return value

    def output_png_path(self, root_id: int) -> Path:
        output_dir_raw = self.output_dir_var.get().strip()
        if output_dir_raw:
            output_dir = Path(output_dir_raw).expanduser()
            if not output_dir.is_absolute():
                output_dir = (Path.cwd() / output_dir).resolve()
        else:
            pkl_path = Path(self.pkl_path_var.get()).expanduser()
            if not pkl_path.is_absolute():
                pkl_path = (Path.cwd() / pkl_path).resolve()
            output_dir = pkl_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"lineage_tree_cell_{root_id}.png"

    def plot_and_save(self) -> None:
        try:
            self.ensure_data_loaded()
            if self.all_cells is None:
                raise RuntimeError("Data did not load correctly.")

            root_id = self.parse_root_id()
            time_steps = self.parse_time_steps()
            max_cells = self.parse_max_cells()

            self.status_var.set(f"Tracing lineage for root cell {root_id}...")
            self.root.update_idletasks()

            full_lineage = trace_lineage(root_id, self.all_cells)
            lineage, was_limited = limit_lineage(full_lineage, max_cells)
            n_nodes, n_alive, n_removed, n_mpp = lineage_summary(lineage)

            self.status_var.set(f"Plotting {n_nodes:,} cells for root cell {root_id}...")
            self.root.update_idletasks()

            draw_lineage_tree(self.figure, lineage, root_id, time_steps, DEFAULT_TIME_UNIT)
            self.canvas.draw_idle()
            self.root.update_idletasks()

            png_path = self.output_png_path(root_id)
            self.figure.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")

            limited_note = " Truncated to max cells." if was_limited else ""
            self.status_var.set(
                f"Saved {png_path}. Root {root_id}: {n_nodes:,} plotted cells, "
                f"{n_alive:,} alive, {n_removed:,} removed, {n_mpp:,} MPP.{limited_note}"
            )

        except Exception as exc:
            traceback.print_exc()
            self.status_var.set("Error: " + str(exc))
            messagebox.showerror("Lineage tree error", str(exc))


def run_gui() -> None:
    root = tk.Tk()
    app = LineageTreeApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
