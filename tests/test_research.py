"""Optional dependency tests: stdlib CI skips them when torch is absent."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(x) for x in ("torch", "numpy", "sklearn"))


@unittest.skipUnless(AVAILABLE, "optional research dependencies not installed")
class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_deterministic_targets_and_mask(self):
        import torch
        from rootforge_skill_surgery.research.tasks import targets
        x = torch.tensor([[0, 9, 10, 2, 4, 6, 8, 15]])
        self.assertEqual(targets("copy", x)[0][0].tolist(), x[0].tolist())
        self.assertEqual(targets("reverse", x)[0][0].tolist(), x[0].flip(0).tolist())
        self.assertEqual(targets("sort", x)[0][0].tolist(), sorted(x[0].tolist()))
        y, mask = targets("count_gt8", x)
        self.assertEqual(int(y[0, 0]), 3)
        self.assertEqual(mask[0].tolist(), [True] + [False] * 7)

    def test_full_patch_is_exact(self):
        import torch
        from rootforge_skill_surgery.research.model import SkillMLP
        from rootforge_skill_surgery.research.tasks import fixed_inputs
        torch.manual_seed(71)
        model = SkillMLP(hidden=16).eval()
        a, b = fixed_inputs(8, 101), fixed_inputs(8, 102)
        with torch.no_grad():
            original, acts = model(a, capture=True)
            for site in ("hidden1", "hidden2"):
                patched = model(b, patch={"site": site, "values": acts[site]})
                torch.testing.assert_close(patched, original, atol=1e-6, rtol=1e-6)

    def test_wrong_shape_patch_rejected(self):
        import torch
        from rootforge_skill_surgery.research.model import SkillMLP
        from rootforge_skill_surgery.research.tasks import fixed_inputs
        model = SkillMLP(hidden=12)
        with self.assertRaisesRegex(ValueError, "shape"):
            model(fixed_inputs(2), patch={"site": "hidden2", "values": torch.ones(2, 3)})

    def test_features_are_finite_and_init_seed_identical(self):
        import torch
        from rootforge_skill_surgery.research.features import describe_model
        from rootforge_skill_surgery.research.model import SkillMLP
        from rootforge_skill_surgery.research.experiment import seed_everything
        seed_everything(345)
        f1 = describe_model(SkillMLP(hidden=16), torch.device("cpu"))
        seed_everything(345)
        f2 = describe_model(SkillMLP(hidden=16), torch.device("cpu"))
        self.assertTrue(any(k.startswith("spectral.") for k in f1))
        self.assertTrue(any(k.startswith("activation.") for k in f1))
        self.assertEqual(f1, f2)

    def test_seed_disjoint_classifier_grouping(self):
        from rootforge_skill_surgery.research.experiment import heldout_seed_prediction
        from rootforge_skill_surgery.research.tasks import TASKS
        # Feature values encode different seed effects but initialization
        # descriptors are identical for all tasks inside each seed.
        rows = []
        for seed in range(3):
            for i, task in enumerate(TASKS):
                rows.append({"seed": seed, "task": task,
                             "eval_accuracy": .6+i*.1,
                             "raw.one": i+seed*.2,
                             "spectral.one": i*.2+seed*.1,
                             "activation.one": i*.1+seed*.3,
                             "init_raw.one": seed*1.4,
                             "init_spectral.one": seed*.3})
        res = heldout_seed_prediction(rows)
        self.assertEqual(res["sample_count"], 12)
        self.assertEqual(res["seed_count"], 3)
        self.assertAlmostEqual(res["chance_balanced"], .25)
        self.assertEqual(len(res["predictions"]), 12*6)

    def test_small_end_to_end_smoke(self):
        from rootforge_skill_surgery.research.experiment import run
        with tempfile.TemporaryDirectory() as dirname:
            output = Path(dirname)/"study"
            report = run(seeds=2, steps=3, batch_size=8, hidden=8,
                         lr=.003, device_name="cpu", output=output,
                         patch_task="copy", patch_top_k=2)
            self.assertEqual(report["sample_count"], 8)
            self.assertTrue((output/"summary.json").exists())
            self.assertTrue((output/"causal_routes.csv").exists())
            self.assertEqual(len(report["patching"]), 2)
            # The three-step smoke is not a capability-removal experiment.
            self.assertIn(report["patching"][0]["status"],
                          ("measured", "insufficient_correct_pairs"))


if __name__ == "__main__":
    unittest.main()
