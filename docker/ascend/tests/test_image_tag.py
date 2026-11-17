"""Exercise product identity and invalid release names without registry access."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("image_tag", Path(__file__).parents[1] / "image_tag.py")
tag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tag_module)


class ImageTagTest(unittest.TestCase):
    def recipe(self, root: Path, base_product: str, *, argument: bool = False) -> Path:
        base = f"quay.io/ascend/cann:9.2.1-{base_product}-ubuntu22.04-py3.12"
        header = f"ARG BASE_IMAGE={base}\nFROM ${{BASE_IMAGE}}" if argument else f"FROM {base}"
        path = root / "Dockerfile"
        path.write_text(header + '\nRUN pip install "torch-npu==2.10.0.post7"\n')
        return path

    def test_product_tags_and_base_argument(self):
        for base, product in [("910b", "910b"), ("a3", "a3"), ("950", "a5")]:
            for argument in [False, True]:
                with self.subTest(product=product, argument=argument), tempfile.TemporaryDirectory() as directory:
                    path = self.recipe(Path(directory), base, argument=argument)
                    self.assertEqual(
                        tag_module.image_tag(path, "v0.1.12", product),
                        f"v0.1.12-cann9.2.1-torch_npu2.10.0.post7-{product}-ubuntu22.04-py3.12-veomni",
                    )

    def test_rejects_wrong_product_and_invalid_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.recipe(Path(directory), "a3")
            with self.assertRaises(ValueError):
                tag_module.image_tag(path, "v0.1.12", "a5")
            for version in ["release/main", "v1+build", "a" * 128, "$(id)"]:
                with self.subTest(version=version), self.assertRaises(ValueError):
                    tag_module.image_tag(path, version, "a3")

    def test_requires_pinned_pta(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.recipe(Path(directory), "a3")
            path.write_text(path.read_text().replace("torch-npu==2.10.0.post7", "torch-npu"))
            with self.assertRaises(ValueError):
                tag_module.image_tag(path, "v0.1.12", "a3")


if __name__ == "__main__":
    unittest.main()
