"""Plot saved Busch labeling results with explicit observable and time labels."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .busch_common import OBSERVATIONS, load_results, read_json, sha256, write_json


def plot(input_path, output_stem, data_path=OBSERVATIONS, after_day=30, time_origin=0):
    result, data = load_results(input_path), read_json(data_path)
    days = result["time_points"]
    mask = days > after_day
    observed_days = np.asarray(data["days"])
    observed_mask = observed_days > after_day
    if not np.any(mask):
        raise ValueError("No simulated times remain after the cutoff.")
    x = days[mask] - time_origin
    lt, st = result["lt_fraction"][:, mask], result["st_labeled_count_ratio"][:, mask]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for ax in axes:
        ax.grid(alpha=0.25)
        for row in lt:
            ax.plot(x, row, color="lightgray", alpha=0.25, linewidth=0.6)
        ax.plot(x, lt.mean(0), "r-", label="LT labeled fraction")
        ax.plot(observed_days[observed_mask] - time_origin,
                np.asarray(data["observations"]["LT-HSC"])[observed_mask],
                "bo", label="LT observations from archive")
    for row in st:
        axes[0].plot(x, row, color="lightgray", alpha=0.25, linewidth=0.6)
    axes[0].plot(x, st.mean(0), "r--", label="ST labeled-count ratio")
    axes[0].plot(observed_days[observed_mask] - time_origin,
                 np.asarray(data["observations"]["ST-HSC"])[observed_mask],
                 "gs", label="ST observations from archive")
    axes[0].set_ylabel("Labeling observable")
    axes[1].set_ylabel("LT labeled fraction")
    axes[1].set_xlabel("Original model update index (days)" if time_origin == 0
                       else f"Days relative to model index {time_origin:g}")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.text(0.5, 0.01,
             "ST model quantity = labeled ST count / labeled LT count. Experimental ST normalization remains unverified.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=200)
    plt.close(fig)
    write_json(output_stem.with_suffix(".json"), {
        "source_results_sha256": sha256(input_path), "data_sha256": sha256(data_path),
        "after_day_exclusive": after_day, "time_origin": time_origin,
        "number_of_seed_rows": len(lt), "st_observable": "labeled ST count / labeled LT count",
        "experimental_st_observable_verified": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-stem", type=Path, default=Path("results/busch_labeling"))
    parser.add_argument("--data", type=Path, default=OBSERVATIONS)
    parser.add_argument("--after-day", type=float, default=30)
    parser.add_argument("--time-origin", type=float, default=0)
    args = parser.parse_args()
    plot(args.input, args.output_stem, args.data, args.after_day, args.time_origin)


if __name__ == "__main__":
    main()
