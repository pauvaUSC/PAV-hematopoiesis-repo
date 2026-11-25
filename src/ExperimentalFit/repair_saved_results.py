"""Recover equally weighted seeds after verifying the original triangular duplication."""
import argparse
import io
import math
import pickle
from pathlib import Path

import numpy as np

try:
    from numpy._core import multiarray
except ImportError:  # NumPy 1.x legacy numeric files.
    from numpy.core import multiarray

from .busch_common import save_results, sha256, write_json


class NumericUnpickler(pickle.Unpickler):
    """Allow only the NumPy array constructors needed by the supplied numeric file."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): multiarray.scalar,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported pickle type: {module}.{name}")
        return allowed[module, name]


def recover_rows(lt, st):
    lt, st = np.asarray(lt, float), np.asarray(st, float)
    if lt.ndim != 2 or lt.shape != st.shape or not np.isfinite(lt).all() or not np.isfinite(st).all():
        raise ValueError("Expected matching finite two-dimensional trajectory arrays.")
    rows = len(lt)
    n = (math.isqrt(1 + 8 * rows) - 1) // 2
    if n < 2 or n * (n + 1) // 2 != rows:
        raise ValueError("No identifiable triangular row count; do not guess seed membership.")
    # Original loop appends prefixes [0], [0,1], ..., [0,...,n-1].
    for matrix in (lt, st):
        last = matrix[-n:]
        offset = 0
        for length in range(1, n + 1):
            if not np.array_equal(matrix[offset:offset + length], last[:length]):
                raise ValueError("Rows do not match the original repeated-prefix bug.")
            offset += length
    return lt[-n:].copy(), st[-n:].copy()


def repair(source, output):
    source, output = Path(source), Path(output)
    raw = NumericUnpickler(io.BytesIO(source.read_bytes())).load()
    old_lt = np.asarray(raw["all_frac_lt_hsc_labeled"], float)
    old_st = np.asarray(raw["all_frac_st_wrt_lt_labeled"], float)
    lt, st = recover_rows(old_lt, old_st)
    save_results(output, raw["time_points"], list(range(len(lt))), lt, st)
    report = {
        "source_sha256": sha256(source), "original_rows": len(old_lt),
        "recovered_unique_seed_rows": len(lt), "pattern_verified": True,
        "seed_order": "0..n-1 inferred from the verified uploaded generator loop",
        "max_absolute_mean_change_lt_fraction": float(np.max(np.abs(old_lt.mean(0) - lt.mean(0)))),
        "max_absolute_mean_change_st_count_ratio": float(np.max(np.abs(old_st.mean(0) - st.mean(0)))),
        "experimental_data": {"days": np.asarray(raw["sample_daysexp"]).tolist(),
                              "observations": raw["experimental_data"]},
        "st_observable": "labeled ST count / labeled LT count, unchanged",
        "experimental_st_observable_verified": False,
        "model_version": "Legacy saved trajectories; not rerun with v6",
        "parameter_provenance": "The old results file does not record model parameters or its code version.",
        "scope": "Corrects aggregation only; cannot recover cell counts or change label inheritance.",
    }
    write_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/busch_saved_repaired.npz"))
    args = parser.parse_args()
    report = repair(args.input, args.output)
    print(f"Verified {report['original_rows']} rows; recovered {report['recovered_unique_seed_rows']} equally weighted seeds.")
    print(args.output)


if __name__ == "__main__":
    main()
