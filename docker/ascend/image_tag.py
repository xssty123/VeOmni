"""Generate the fork's product-based Quay tag from an existing Dockerfile."""

import argparse
import re
from pathlib import Path


def image_tag(dockerfile: Path, version: str, product: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", version):
        raise ValueError("Version must be a Docker tag component")
    text = dockerfile.read_text()
    base = re.search(r"^FROM\s+(\S+)", text, re.MULTILINE)
    if base is None:
        raise ValueError("Dockerfile has no FROM image")
    base_image = base.group(1)
    if base_image == "${BASE_IMAGE}":
        default = re.search(r"^ARG BASE_IMAGE=(\S+)", text, re.MULTILINE)
        if default is None:
            raise ValueError("BASE_IMAGE has no default")
        base_image = default.group(1)
    base_tag = base_image.rsplit(":", 1)[-1]
    match = re.fullmatch(r"([^-]+)-(910b|a3|a5|950)-(ubuntu[0-9.]+)-(py[0-9.]+)", base_tag)
    if match is None:
        raise ValueError(f"Unsupported CANN base tag: {base_tag}")
    cann, base_product, os_name, python = match.groups()
    expected_product = "a5" if base_product == "950" else base_product
    if product != expected_product:
        raise ValueError(f"Product {product} does not match base product {base_product}")
    pta = re.search(r"torch-npu==([0-9A-Za-z.+]+)", text)
    if pta is None:
        raise ValueError("Dockerfile has no pinned torch-npu version")
    tag = f"{version}-cann{cann}-torch_npu{pta.group(1)}-{product}-{os_name}-{python}-veomni"
    if len(tag) > 128 or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", tag):
        raise ValueError("Generated image tag is invalid")
    return tag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockerfile", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--product", required=True, choices=("910b", "a3", "a5"))
    args = parser.parse_args()
    print(image_tag(args.dockerfile, args.version, args.product))


if __name__ == "__main__":
    main()
