"""Preserve the original Busch optimization method; add saved evaluation records."""
import argparse
import time
from pathlib import Path

import numpy as np

from .busch_common import (CONFIG, OBSERVATIONS, FIT_NAMES, labeling_observables,
                           load_inputs, parameters, provenance, run_counts,
                           validate_parameters, write_json)

BOUNDS = [(0.01, 0.5), (0.01, 0.5), (0.01, 0.4), (0.01, 0.4),
          (2, 20), (2, 20), (5, 20), (2, 10)]


def candidate_parameters(values, baseline):
    p = baseline.copy()
    p.update(zip(FIT_NAMES, map(float, values), strict=True))
    for suffix in ("A", "B"):
        p[f"P4{suffix}"] = 1 - p[f"P1{suffix}"] - p[f"P2{suffix}"] - p[f"P3{suffix}"]
    validate_parameters(p)
    return p


def evaluate(p, config, data):
    curves = []
    for seed in config["seeds"]:
        _, curve = labeling_observables(run_counts(config, p, seed))
        curves.append(curve)
    mean = np.mean(curves, axis=0)
    loss = float(np.sum((mean - data["observations"]["ST-HSC"]) ** 2))
    if not np.isfinite(loss):
        raise ValueError("Nonfinite objective value.")
    return loss, mean.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--data", type=Path, default=OBSERVATIONS)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/busch_fit"))
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Score existing parameters without optimization or scikit-optimize.")
    args = parser.parse_args()
    config, data = load_inputs(args.config, args.data)
    if not 1 <= args.seed_count <= len(config["seeds"]):
        parser.error("seed-count must fit the configured seed list")
    config = dict(config, sample_days=data["days"], time_steps=max(data["days"]) + 5,
                  seeds=config["seeds"][:args.seed_count])
    p = parameters(args.parameters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup = provenance(config, p, data, fitting={
        "objective": "SSE of the mean per-seed labeled ST count / labeled LT count",
        "fit_names": FIT_NAMES, "bounds": BOUNDS, "dimensions": "Real for all eight, unchanged",
        "n_calls": args.calls, "random_state": 42,
        "optimizer": "skopt.gp_minimize with original unspecified defaults",
        "scope": "No change to labeling biology, objective, search bounds, or threshold dimensions.",
    })
    if args.evaluate_only:
        loss, curve = evaluate(p, config, data)
        write_json(args.output_dir / "evaluation.json", {**setup, "loss": loss, "mean_st_curve": curve})
        print(f"Existing-parameter ST loss: {loss:.8g}")
        return
    try:
        import skopt
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError as exc:
        raise SystemExit("Install scikit-optimize to refit; --evaluate-only does not require it.") from exc
    if args.calls < 10:
        parser.error("The unchanged gp_minimize default needs at least 10 calls.")
    setup["fitting"]["scikit_optimize_version"] = skopt.__version__
    write_json(args.output_dir / "setup.json", setup)
    history = []

    def objective(values):
        start = time.perf_counter()
        record = {"candidate": list(map(float, values))}
        try:
            trial = candidate_parameters(values, p)
            loss, curve = evaluate(trial, config, data)
            record.update(parameters=trial, mean_st_curve=curve, status="ok")
        except ValueError as exc:
            # Invalid probabilities and undefined labeling ratios are explicit penalties.
            loss = 1e6
            record.update(status="invalid_candidate_or_prediction", reason=str(exc))
        record.update(loss=loss, elapsed_seconds=time.perf_counter() - start)
        history.append(record)
        write_json(args.output_dir / "history.json", history)
        return loss

    result = gp_minimize(objective, [Real(*bounds, name=name) for bounds, name in zip(BOUNDS, FIT_NAMES)],
                         n_calls=args.calls, random_state=42, verbose=True)
    if not any(record["status"] == "ok" for record in history):
        raise RuntimeError("No valid fit evaluations; inspect history.json.")
    best = candidate_parameters(result.x, p)
    write_json(args.output_dir / "best_parameters.json", {
        "parameters": best, "loss": float(result.fun),
        "candidate_history": result.x_iters, "loss_history": result.func_vals.tolist(),
        "model_sha256": setup["model_sha256"],
        "uncertainty_note": "Optimization output is not a parameter posterior or confidence interval.",
    })
    print(args.output_dir / "best_parameters.json")


if __name__ == "__main__":
    main()
