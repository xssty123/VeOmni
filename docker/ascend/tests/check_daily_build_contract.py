"""Validate the reproducible A5 daily-wheel build inputs without accelerator imports."""

import argparse
import json
import re
from importlib.metadata import version
from pathlib import Path


def check_installed(profile):
    observed = {name: version(name) for name in profile["versions"]}
    for name, expected in profile["versions"].items():
        if observed[name].split("+")[0] != expected:
            raise RuntimeError(f"Unexpected {name}: {observed[name]}, expected {expected}")
    fla = profile["fla"]
    observed[fla["distribution"]] = version(fla["distribution"])
    if observed[fla["distribution"]] != fla["version"]:
        raise RuntimeError(f"Unexpected FLA daily: {observed[fla['distribution']]}")
    return observed


def check_sources(profile, root):
    dockerfile = (root / profile["dockerfile"]).read_text()
    if f"FROM {profile['base_image']}\n" not in dockerfile:
        raise RuntimeError("Base tag/digest differs from the selected profile")
    for name in ("fla", "triton_ascend", "fla_checker"):
        artifact = profile[name]
        if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
            raise RuntimeError(f"Invalid {name} SHA256")
        if artifact["url"] not in dockerfile or artifact["sha256"] not in dockerfile:
            raise RuntimeError(f"Dockerfile does not pin the selected {name} URL/hash")
    if "/latest/" in profile["fla"]["url"] or "manylinux_2_34_aarch64.whl" not in profile["fla"]["filename"]:
        raise RuntimeError("FLA must use a specific ARM64 artifact, not a moving directory")
    if profile["fla"]["version"] not in profile["fla"]["filename"]:
        raise RuntimeError("FLA wheel filename and version disagree")
    if profile["torchcodec"]["source_commit"] not in dockerfile:
        raise RuntimeError("TorchCodec source is not pinned")
    if any(token in dockerfile for token in ("build_wheel.py", "FLA_BUILD_JOBS", "AS fla-build-env")):
        raise RuntimeError("Daily-wheel path must not compile FLA or expose source-build stages")
    if 'pip install -e ".[npu_aarch64]" "transformers==5.16.1"' not in dockerfile:
        raise RuntimeError("ARM extra/Transformers installation contract changed")
    for package in ("torch-npu", "triton", "pytest", "expecttest", "rich", "pyarrow", "pandas"):
        if f"{package}=={profile['versions'][package]}" not in dockerfile:
            raise RuntimeError(f"Dockerfile differs from selected {package} version")
    print(f"Build contract passed: {profile['name']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    profile = json.loads((root / "docker/ascend/profiles/a5-daily-20261111.json").read_text())
    check_sources(profile, root)
    if args.installed:
        print(json.dumps(check_installed(profile), indent=2))


if __name__ == "__main__":
    main()
