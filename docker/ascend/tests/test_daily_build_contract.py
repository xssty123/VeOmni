"""CPU-only regression checks for the A5 daily-wheel build contract."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import check_daily_build_contract as contract


ROOT = Path(__file__).resolve().parents[3]
PROFILE = json.loads((ROOT / "docker/ascend/profiles/a5-daily-20261111.json").read_text())


class DailyBuildContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = copy.deepcopy(PROFILE)

    def check(self, profile=None, root=ROOT):
        with contextlib.redirect_stdout(io.StringIO()):
            contract.check_sources(profile or self.profile, root)

    def test_selected_inputs_match(self):
        self.check()

    def test_changed_artifact_hash_is_rejected(self):
        for name in ("fla", "triton_ascend", "fla_checker"):
            with self.subTest(name=name):
                profile = copy.deepcopy(PROFILE)
                profile[name]["sha256"] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, "URL/hash"):
                    self.check(profile)

    def test_changed_base_digest_is_rejected(self):
        self.profile["base_image"] += "0"
        with self.assertRaisesRegex(RuntimeError, "Base tag/digest"):
            self.check()

    def test_changed_torchcodec_source_is_rejected(self):
        self.profile["torchcodec"]["source_commit"] = "0" * 40
        with self.assertRaisesRegex(RuntimeError, "TorchCodec source"):
            self.check()

    def test_fla_source_compilation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dockerfile = root / self.profile["dockerfile"]
            dockerfile.parent.mkdir(parents=True)
            dockerfile.write_text((ROOT / self.profile["dockerfile"]).read_text() + "\nRUN python build_wheel.py\n")
            with self.assertRaisesRegex(RuntimeError, "must not compile FLA"):
                self.check(root=root)

    def test_selected_installed_versions_are_accepted(self):
        versions = {**self.profile["versions"], self.profile["fla"]["distribution"]: self.profile["fla"]["version"]}
        versions["torchcodec"] += "+0b261b9"
        with patch.object(contract, "version", side_effect=versions.__getitem__):
            self.assertEqual(contract.check_installed(self.profile), versions)

    def test_other_daily_version_is_rejected(self):
        versions = {**self.profile["versions"], self.profile["fla"]["distribution"]: "0.dev20261110+main.other"}
        with patch.object(contract, "version", side_effect=versions.__getitem__):
            with self.assertRaisesRegex(RuntimeError, "Unexpected FLA daily"):
                contract.check_installed(self.profile)

    def test_changed_runtime_version_is_rejected(self):
        versions = {**self.profile["versions"], "triton-ascend": "3.6.0"}
        with patch.object(contract, "version", side_effect=versions.__getitem__):
            with self.assertRaisesRegex(RuntimeError, "Unexpected triton-ascend"):
                contract.check_installed(self.profile)


if __name__ == "__main__":
    unittest.main()
