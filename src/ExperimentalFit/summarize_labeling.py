"""Report plateau and first-crossing summaries without silently shifting the clock."""
import argparse
from pathlib import Path

import numpy as np

from .busch_common import load_results, write_json


def summaries(result, after_day=30, plateau_from=200):
    t = result["time_points"]
    mask = t > after_day
    plateau_mask = t >= plateau_from
    if not np.any(mask) or not np.any(plateau_mask):
        raise ValueError("The requested analysis and plateau windows must contain samples.")
    st = result["st_labeled_count_ratio"]
    plateau = st[:, plateau_mask].mean(1)
    crossings = []
    for curve, value in zip(st[:, mask], plateau):
        indices = np.flatnonzero(curve >= 0.5 * value)
        crossings.append(float(t[mask][indices[0]]) if value > 0 and len(indices) else None)
    finite = np.asarray([x for x in crossings if x is not None])
    return {
        "seed_count": len(st), "after_day_exclusive": after_day, "plateau_from": plateau_from,
        "plateau_per_seed": plateau.tolist(), "plateau_mean": float(plateau.mean()),
        "plateau_sd_ddof0_legacy": float(plateau.std(ddof=0)),
        "plateau_sd_ddof1": float(plateau.std(ddof=1)) if len(plateau) > 1 else None,
        "first_half_plateau_crossing_model_day_per_seed": crossings,
        "crossing_mean_model_day": float(finite.mean()) if len(finite) else None,
        "crossing_sd_model_day_ddof0": float(finite.std()) if len(finite) else None,
        "crossing_missing_seed_count": sum(x is None for x in crossings),
        "note": "Crossings use sampled original model days, not elapsed residence times. No automatic tau conversion is made.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/busch_summary.json"))
    parser.add_argument("--after-day", type=float, default=30)
    parser.add_argument("--plateau-from", type=float, default=200)
    args = parser.parse_args()
    value = summaries(load_results(args.input), args.after_day, args.plateau_from)
    write_json(args.output, value)
    print(f"Plateau mean {value['plateau_mean']:.6g}; {value['seed_count']} equally weighted seeds")


if __name__ == "__main__":
    main()
