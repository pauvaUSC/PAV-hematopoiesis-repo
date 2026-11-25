"""Score archived, division-fix-only, and v6 models with identical fixed parameters."""
import argparse
import hashlib
import types
from pathlib import Path

import numpy as np

from .busch_common import (CONFIG, OBSERVATIONS, labeling_observables, load_inputs,
                           model, parameters, run_counts, write_json)


def load_source(source, name):
    engine = types.ModuleType(name)
    exec(compile(source, name, "exec"), engine.__dict__)
    return engine


def compare(source, config, p, data):
    needle = '    if cell.division_count >= division_threshold:\n'
    replacement = '    if cell.state != "Apoptotic" and cell.division_count >= division_threshold:\n'
    if source.count(needle) != 1:
        raise ValueError("Baseline does not have the expected division-limit statement.")
    engines = {
        "archive_original": load_source(source, "archive_original"),
        "division_fix_only": load_source(source.replace(needle, replacement), "division_fix_only"),
        "v6": model,
    }
    curves, sampled_counts = {}, {}
    for name, engine in engines.items():
        traces = []
        sampled_counts[name] = []
        for seed in config["seeds"]:
            counts = run_counts(config, p, seed, engine)
            _, st = labeling_observables(counts)
            traces.append(st)
            sampled_counts[name].append(counts)
            print(f"Completed {name}, seed {seed}", flush=True)
        curves[name] = np.asarray(traces)
    result = {
        "config": config, "parameters": p, "experimental_data": data,
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "st_observable": "labeled ST count / labeled LT count, unchanged",
        "scores": {}, "comparison_note": "v6 also removes duplicate initialization, which shifts RNG draws relative to the archive. The division-fix-only control isolates the removal correction.",
    }
    for name, curve in curves.items():
        changed = set()
        for before, after in zip(sampled_counts["archive_original"], sampled_counts[name]):
            changed.update(k for k in before if before[k] != after[k])
        result["scores"][name] = {
            "st_sse_of_mean": float(np.sum((curve.mean(0) - data["observations"]["ST-HSC"]) ** 2)),
            "st_curves": curve.tolist(), "mean_st_curve": curve.mean(0).tolist(),
            "changed_sampled_count_fields_vs_archive": sorted(changed),
            "max_absolute_st_curve_change_vs_archive": float(np.max(np.abs(curve - curves["archive_original"]))),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", type=Path, default=Path("src/ExperimentalFit/model_test.py"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--data", type=Path, default=OBSERVATIONS)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/busch_model_comparison.json"))
    args = parser.parse_args()
    config, data = load_inputs(args.config, args.data)
    if not 1 <= args.seed_count <= len(config["seeds"]):
        parser.error("seed-count must fit the configured seed list")
    config = dict(config, sample_days=data["days"], time_steps=max(data["days"]) + 5,
                  seeds=config["seeds"][:args.seed_count])
    result = compare(args.baseline_model.read_text(), config, parameters(args.parameters), data)
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
