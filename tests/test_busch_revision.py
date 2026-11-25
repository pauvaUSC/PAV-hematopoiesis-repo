import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ExperimentalFit.busch_common import (FIT_NAMES, labeling_observables, model,
                                             save_results, load_results)
from src.ExperimentalFit.fit_labeling import candidate_parameters
from src.ExperimentalFit.repair_saved_results import NumericUnpickler, recover_rows
from src.ExperimentalFit.summarize_labeling import summaries


class FixedRng:
    def random(self):
        return 0.5

    def normal(self, mean, std):
        return mean


class BuschRevisionTests(unittest.TestCase):
    def test_all_fates_below_and_at_limit(self):
        for typ, suffix in (("LT-HSC", "A"), ("ST-HSC", "B")):
            for fate in range(1, 5):
                for at_limit in (False, True):
                    with self.subTest(typ=typ, fate=fate, at_limit=at_limit):
                        p = model.params.copy()
                        for k in range(1, 5):
                            p[f"P{k}{suffix}"] = float(k == fate)
                        cap = p[f"{typ.replace('-', '_')}_DIVISION_THRESHOLD"]
                        rng = FixedRng()
                        cell = model.Cell(typ, "Non-quiescent Active", 0, 999999, p, rng, rng, rng,
                                          division_count=cap - (1 if at_limit else 2), label_intensity=1)
                        daughters = []
                        model.perform_division(cell, daughters, 100, [])
                        expected = "Apoptotic" if fate >= 3 else (
                            "Non-quiescent Inactive" if at_limit else "Non-quiescent Active")
                        self.assertEqual(cell.state, expected)
                        self.assertEqual(len(daughters), 2 if fate == 4 else 1)
                        self.assertTrue(all(d.label_intensity == 0.5 for d in daughters))

    def test_differentiated_parent_is_removed_from_simulation(self):
        for typ, suffix in (("LT-HSC", "A"), ("ST-HSC", "B")):
            for fate in (3, 4):
                p = model.params.copy()
                p.update(PQA=0, PQB=0, PAA=0, PAB=0, LABELING_FRACTION=0, LABELING_PROB=0)
                for k in range(1, 5):
                    p[f"P{k}{suffix}"] = float(k == fate)
                cap = p[f"{typ.replace('-', '_')}_DIVISION_THRESHOLD"]
                rng = FixedRng()
                cell = model.Cell(typ, "Non-quiescent Active", 0, 999999, p, rng, rng, rng,
                                  division_count=cap - 1, time_to_divide=1)
                counts, cells, history, _, _ = model.simulate_hematopoiesis(
                    0, 0, p, 1, [0], 0, 0, seed=1, initial_cells=[cell])
                self.assertNotIn(cell, cells)
                self.assertEqual(counts[f"{typ}_nonquiescent_inactive"], [0])
                self.assertEqual(counts[f"{typ}_removed"], [1])
                self.assertEqual(history[0]["end_reason"], "apoptosis_due_to_differentiation")

    def test_recover_triangular_results_not_value_deduplication(self):
        # Distinct seeds can legitimately produce identical trajectories.
        unique = np.array([[1., 2.], [1., 2.], [3., 4.]])
        old = np.concatenate([unique[:n] for n in (1, 2, 3)])
        lt, st = recover_rows(old, old * 2)
        np.testing.assert_array_equal(lt, unique)
        np.testing.assert_array_equal(st, unique * 2)
        broken = old.copy()
        broken[1, 0] = 123
        with self.assertRaises(ValueError):
            recover_rows(broken, old)

    def test_count_ratio_and_lt_fraction_are_explicit(self):
        counts = {"LT-HSC_active_labeled": [2], "LT-HSC_quiescent_labeled": [0],
                  "LT-HSC_active_unlabeled": [8], "LT-HSC_quiescent_unlabeled": [0],
                  "ST-HSC_active_labeled": [3], "ST-HSC_quiescent_labeled": [0]}
        lt, st = labeling_observables(counts)
        self.assertEqual(lt.tolist(), [0.2])
        self.assertEqual(st.tolist(), [1.5])
        counts["LT-HSC_active_labeled"] = [0]
        with self.assertRaises(ValueError):
            labeling_observables(counts)

    def test_invalid_probabilities_rejected_without_mutation(self):
        p = model.params.copy()
        values = [p[k] for k in FIT_NAMES]
        values[:2] = [0.5, 0.5]
        with self.assertRaises(ValueError):
            candidate_parameters(values, p)
        self.assertEqual(p, model.params)

    def test_saved_format_has_one_row_per_seed_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.npz"
            save_results(path, [0, 7], [10, 11], [[1, 2], [3, 4]], [[0, 1], [1, 2]])
            self.assertEqual(load_results(path)["seeds"].tolist(), [10, 11])
            with self.assertRaises(ValueError):
                save_results(path, [0, 7], [10, 10], [[1, 2], [3, 4]], [[0, 1], [1, 2]])

    def test_summary_preserves_original_clock_and_missing_crossings(self):
        result = {"time_points": np.array([0, 35, 70, 210]),
                  "st_labeled_count_ratio": np.array([[0, .1, .3, .4], [0, 0, 0, 0]])}
        value = summaries(result)
        self.assertEqual(value["first_half_plateau_crossing_model_day_per_seed"], [70, None])
        self.assertEqual(value["crossing_missing_seed_count"], 1)

    def test_restricted_loader_rejects_non_numeric_globals(self):
        import io
        with self.assertRaises(pickle.UnpicklingError):
            NumericUnpickler(io.BytesIO(pickle.dumps(Path("unneeded")))).load()


if __name__ == "__main__":
    unittest.main()
