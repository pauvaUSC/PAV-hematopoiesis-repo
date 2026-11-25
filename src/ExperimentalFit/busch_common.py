"""Shared I/O and unchanged legacy Busch labeling observables."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from src.models import hematopoiesis_model_v6 as model

DATA = Path(__file__).with_name("data")
OBSERVATIONS = DATA / "busch_2015_observations.json"
CONFIG = DATA / "busch_2015_config.json"
FIT_NAMES = ["P2A", "P3A", "P2B", "P3B", "LT_HSC_DIVISION_THRESHOLD",
             "ST_HSC_DIVISION_THRESHOLD", "LT_HSC_TIME_STD", "ST_HSC_TIME_STD"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(config_path=CONFIG, data_path=OBSERVATIONS):
    config, data = read_json(config_path), read_json(data_path)
    days = np.asarray(data["days"], dtype=float)
    if days.ndim != 1 or not len(days) or not np.isfinite(days).all():
        raise ValueError("Observation days must be a nonempty finite vector.")
    if np.any(days != days.astype(int)) or np.any(np.diff(days) <= 0) or days[0] < 0:
        raise ValueError("Observation days must be increasing nonnegative integers.")
    for kind in ("LT-HSC", "ST-HSC"):
        observed = np.asarray(data["observations"][kind], dtype=float)
        if observed.shape != days.shape or not np.isfinite(observed).all():
            raise ValueError(f"Invalid or mismatched {kind} observations.")
    samples = config["sample_days"]
    if (not samples or samples != sorted(set(samples))
            or any(not isinstance(t, int) or not 0 <= t < config["time_steps"] for t in samples)):
        raise ValueError("Sample days must be unique integers inside range(time_steps).")
    if not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Simulation seeds must be nonempty and unique.")
    for key in ("initial_lt_hsc_count", "initial_lt_quiescent", "initial_st_hsc_count", "initial_st_quiescent"):
        if not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    return config, data


def parameters(path=None):
    result = model.params.copy()
    if path:
        supplied = read_json(path)
        result.update(supplied.get("parameters", supplied))
    validate_parameters(result)
    return result


def validate_parameters(p):
    for suffix in ("A", "B"):
        probs = np.asarray([p[f"P{k}{suffix}"] for k in (1, 2, 3, 4)], float)
        if not np.isfinite(probs).all() or np.any(probs < 0) or np.any(probs > 1) or not np.isclose(probs.sum(), 1):
            raise ValueError(f"Invalid fate probabilities for {suffix}.")
    if any(not np.isfinite(value) for value in p.values()):
        raise ValueError("All model parameters must be finite.")


def labeling_observables(counts):
    labeled_lt = np.asarray(counts["LT-HSC_active_labeled"], float) + counts["LT-HSC_quiescent_labeled"]
    unlabeled_lt = np.asarray(counts["LT-HSC_active_unlabeled"], float) + counts["LT-HSC_quiescent_unlabeled"]
    labeled_st = np.asarray(counts["ST-HSC_active_labeled"], float) + counts["ST-HSC_quiescent_labeled"]
    total_lt = labeled_lt + unlabeled_lt
    if np.any(total_lt <= 0) or np.any(labeled_lt <= 0):
        raise ValueError("The legacy labeling observable has a zero denominator.")
    lt_fraction, st_count_ratio = labeled_lt / total_lt, labeled_st / labeled_lt
    if not np.isfinite(lt_fraction).all() or not np.isfinite(st_count_ratio).all():
        raise ValueError("Nonfinite labeling prediction.")
    return lt_fraction, st_count_ratio


def run_counts(config, p, seed, engine=model):
    # Reset IDs per independent realization; IDs do not affect stochastic draws.
    engine.cell_id_counter = 0
    return engine.simulate_hematopoiesis(
        initial_lt_hsc_count=config["initial_lt_hsc_count"],
        initial_st_hsc_count=config["initial_st_hsc_count"],
        params=p.copy(), time_steps=config["time_steps"], sample_days=config["sample_days"],
        initial_lt_quiescent=config["initial_lt_quiescent"],
        initial_st_quiescent=config["initial_st_quiescent"], seed=seed,
    )[0]


def provenance(config, p, data, **extra):
    return {
        "model": "hematopoiesis_model_v6", "model_sha256": sha256(model.__file__),
        "python": platform.python_version(), "numpy": np.__version__,
        "config": config, "parameters": p, "experimental_data": data,
        "st_observable": "labeled ST count / labeled LT count",
        "lt_observable": "labeled LT / (active + quiescent LT)",
        "experimental_st_observable_verified": False,
        "time_convention": "Original model update indices; no automatic 30-day shift.",
        **extra,
    }


def save_results(path, days, seeds, lt, st):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lt, st = np.asarray(lt, float), np.asarray(st, float)
    shape = (len(seeds), len(days))
    if lt.shape != shape or st.shape != shape or len(set(seeds)) != len(seeds):
        raise ValueError("Results must contain exactly one row per unique seed.")
    if not np.isfinite(lt).all() or not np.isfinite(st).all():
        raise ValueError("Cannot save nonfinite results.")
    np.savez_compressed(path, time_points=days, seeds=seeds,
                        lt_fraction=lt, st_labeled_count_ratio=st)


def load_results(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    lt, st = result["lt_fraction"], result["st_labeled_count_ratio"]
    if lt.shape != st.shape or lt.shape != (len(result["seeds"]), len(result["time_points"])):
        raise ValueError("Inconsistent saved result dimensions.")
    if len(set(result["seeds"].tolist())) != len(result["seeds"]):
        raise ValueError("Saved seed identifiers are not unique.")
    if not np.isfinite(lt).all() or not np.isfinite(st).all():
        raise ValueError("Saved predictions must be finite.")
    return result
