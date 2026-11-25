"""Generate one saved labeling trajectory per seed with the corrected model."""
import argparse
from pathlib import Path

from .busch_common import (CONFIG, OBSERVATIONS, labeling_observables, load_inputs,
                           parameters, provenance, run_counts, save_results, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--data", type=Path, default=OBSERVATIONS)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/busch_labeling.npz"))
    args = parser.parse_args()
    config, data = load_inputs(args.config, args.data)
    p = parameters(args.parameters)
    lt, st = [], []
    for seed in config["seeds"]:
        a, b = labeling_observables(run_counts(config, p, seed))
        lt.append(a)
        st.append(b)
        print(f"Completed seed {seed}", flush=True)
    save_results(args.output, config["sample_days"], config["seeds"], lt, st)
    write_json(args.output.with_suffix(".json"), provenance(config, p, data))
    print(args.output)


if __name__ == "__main__":
    main()
