# Busch labeling revision

This workflow repairs the Busch analysis supplied in `Busch_etal_2015.zip` while
preserving the original fitting method for the first controlled comparison.
Run commands from the repository root with Python 3.10 or newer.

## What changed

- All new scripts import the single `src/models/hematopoiesis_model_v6.py` model.
- The division-limit rule cannot turn a differentiated parent back into an inactive HSC.
- v6 also omits the archive model's second population initialization. That fixes
  overwriting supplied initial cells, but changes the RNG position even for a fresh run.
- New ensembles save one row per seed. A verified recovery command repairs the
  archive's repeated-prefix aggregation without rerunning the model.
- Experimental values and initial populations are explicit input files. Outputs
  record parameters, seeds, data, model hash, and time conventions.
- Simulation, optimization, plotting, and summary extraction are separate commands.
  Nothing runs on import. New numerical output uses NPZ with pickle disabled.

## What stays unchanged for this comparison

The ST model observable remains **labeled ST count / labeled LT count**. The LT
observable remains labeled LT divided by active plus quiescent LT. The experimental
values and observation days are copied unchanged from the archive. The archive
does not establish the original experimental figure, digitization, or normalization;
the ST observable therefore still needs biological verification before interpreting
the fit as a validated experimental comparison. No label-inheritance change,
homeostatic feedback, or new quiescence rule has been introduced.

The fitter retains eight parameters, the original `Real` dimensions (including
division thresholds), search bounds, unweighted ST sum of squared errors, five
seeds, 100 `gp_minimize` calls, and optimizer random state 42. This deliberately
does not combine a bug comparison with a change to the fitting method. Any later
change to integer dimensions, experimental weighting, or label biology must be
recorded as a separate analysis.

The archived initialization is 5,000 active plus 15,000 quiescent LT-HSCs and
5 active plus 5 quiescent ST-HSCs. These settings are preserved for comparison,
not established as the best biological initialization. The plotting grid is weekly,
with the eight exact experimental days added so comparisons do not need interpolation.

## Install and test

```bash
python -m pip install -r src/ExperimentalFit/requirements-busch.txt
python -m unittest discover -s tests -p 'test_busch_revision.py' -v
```

`scikit-optimize` is needed only for a new optimization. Recovery, plotting,
fixed-parameter scoring, and comparison do not import it. The revision was checked
with NumPy 2.3.5 and Matplotlib 3.10.8; an optimizer run should also record the
installed scikit-optimize version in its output.

## First recover the existing saved trajectories

```bash
python -m src.ExperimentalFit.repair_saved_results /path/to/simulation_results.pkl --output results/busch_saved_repaired.npz
python -m src.ExperimentalFit.plot_labeling results/busch_saved_repaired.npz --output-stem results/busch_saved_repaired
python -m src.ExperimentalFit.summarize_labeling results/busch_saved_repaired.npz --output results/busch_saved_repaired_summary.json
```

Recovery requires the exact triangular repeated-prefix pattern. It does not remove
rows just because their values happen to be equal: different seeds can legitimately
produce identical trajectories. It recovers the final prefix containing all seeds
and recomputes equal-weight means. The source file remains intact. Repaired output
contains legacy trajectories, **not trajectories generated with v6**; the old file
does not record its generating parameters or code version.

Plotting defaults to original day indices with `t > 30`, matching `plot_both.py`.
`--time-origin 30` explicitly shifts both simulated and observed times by 30 days;
no implicit `+1` adjustment is applied. The summary reports sampled first crossing
days and clearly distinguishes population SD (`ddof=0`, legacy) from sample SD
(`ddof=1`). It does not automatically interpret a crossing day as a residence time.

## Then compare models before refitting

```bash
python -m src.ExperimentalFit.compare_models --baseline-model /path/to/original/model_test.py --seed-count 1 --output results/busch_model_comparison.json
```

In the existing repository, the baseline defaults to `src/ExperimentalFit/model_test.py`.
Use the archived model to make provenance explicit. This command compares:

1. The archive model without changes.
2. An in-memory division-fix-only control that preserves duplicate initialization.
3. The shared v6 model with both corrections.

All use the same parameter values, initial populations, sample days, and seeds.
The control separates the division-limit correction from the changed RNG sequence.
The default parameters are the archived model defaults, not a newly estimated fit.
Pass `--parameters results/busch_fit/best_parameters.json` to use another saved set.
A one-seed check tests mechanics; it is insufficient to assess fitting uncertainty.

## Generate a new ensemble or score existing parameters

```bash
python -m src.ExperimentalFit.run_labeling --output results/busch_labeling.npz
python -m src.ExperimentalFit.plot_labeling results/busch_labeling.npz
python -m src.ExperimentalFit.fit_labeling --evaluate-only --output-dir results/busch_existing_parameter_score
```

The ensemble defaults to 100 seeds. The fixed-parameter score uses the original
five-seed fitting protocol and exact experimental days. To change settings, provide
an explicit JSON `--config` and retain it with the results.

## Refit only after examining the comparison

```bash
python -m src.ExperimentalFit.fit_labeling --output-dir results/busch_fit
```

The run saves setup, each candidate evaluation, reasons for invalid predictions,
best parameters, and the objective history. Unexpected programming exceptions are
not silently converted to penalties. A completed search is not evidence of
convergence or a parameter confidence interval; those require separate checks.

## Files superseded for this workflow

The active commands above replace the combined `plot_plus_run_model_Test_expdata.py`,
`plot_from_saved_data.py`, `cut_plot.py`, `plot_both.py`, `extract_comp_values.py`, and
the old fitter for revised analysis. Keep those files as historical sources until
the revised workflow is accepted. `model_test copy.py` is an identical duplicate
of the archive model and should not be an active implementation. Historical
results must remain identifiable; do not overwrite their names with new outputs.

## Decisions still needed for the manuscript revision

- Verify the experimental labeling quantity and permanent-label interpretation.
- Reconcile calibration and population-simulation initialization and scaling.
- Assess optimization stability, identifiability, and experimental variability.
- Confirm the time origin before interpreting equilibration as a residence time.

These are scientific issues raised by the reviews; changing code organization or
obtaining a smaller fitting error does not by itself resolve them.
