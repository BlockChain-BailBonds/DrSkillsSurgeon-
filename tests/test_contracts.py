import copy
import json
import unittest
from pathlib import Path
from rootforge_skill_surgery.contracts import ContractError, validate_contract

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "toy-contract.json"


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_valid_toy(self):
        result = validate_contract(self.contract)
        self.assertEqual(result["probe"], 2)
        self.assertEqual(result["pairs"], 2)

    def test_experiment_size_gate(self):
        with self.assertRaisesRegex(ContractError, "at least 500"):
            validate_contract(self.contract, minimum_probes=500)

    def test_duplicate_id(self):
        self.contract["retain"][0]["id"] = "p1"
        with self.assertRaisesRegex(ContractError, "duplicate prompt ID"):
            validate_contract(self.contract)

    def test_unpaired_probe(self):
        self.contract["pairs"].pop()
        with self.assertRaisesRegex(ContractError, "unpaired probe IDs"):
            validate_contract(self.contract)

    def test_unknown_contrast(self):
        self.contract["pairs"][0]["contrast_id"] = "does-not-exist"
        with self.assertRaisesRegex(ContractError, "outside its declared set"):
            validate_contract(self.contract)

    def test_empty_prompt(self):
        self.contract["adversarial"][0]["prompt"] = "  "
        with self.assertRaisesRegex(ContractError, "nonempty string"):
            validate_contract(self.contract)

    def test_bad_metric_direction(self):
        self.contract["metric"]["higher_is_more_skill"] = False
        with self.assertRaisesRegex(ContractError, "must be below floor"):
            validate_contract(self.contract)

    def test_bad_minimum(self):
        with self.assertRaisesRegex(ContractError, "positive integer"):
            validate_contract(self.contract, minimum_probes=0)

    def test_duplicate_pair(self):
        self.contract["pairs"].append(copy.deepcopy(self.contract["pairs"][0]))
        with self.assertRaisesRegex(ContractError, "duplicate pair"):
            validate_contract(self.contract)


if __name__ == "__main__":
    unittest.main()
