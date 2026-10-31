"""Check the preinstalled A5 candidate without a venv, uv sync, or NPU execution."""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


FLA_COMMIT = "a7c608f2e3f828779ceeec1fcf155a6e37ede83c"
EXPECTED_VERSIONS = {
    "torch": "2.10.0",
    "torch-npu": "2.10.0.post7",
    "triton": "3.6.0",
    "triton-ascend": "3.6.0",
    "transformers": "5.16.1",
    "torchvision": "0.25.0",
    "torchaudio": "2.10.0",
    # v0.10.0's source version.txt is 0.10.0a0; setup.py appends +<git SHA>.
    "torchcodec": "0.10.0a0",
    "pyarrow": "21.0.0",
    "pandas": "2.3.2",
}


def check_image(report, image_inspect, expected_docker_commit):
    if platform.machine() != "aarch64" or sys.version_info[:3] != (3, 12, 16):
        raise RuntimeError("Expected A5 ARM64 image with Python 3.12.16")
    image = json.loads(image_inspect.read_text())[0]
    labels = image["Config"]["Labels"]
    info = {
        "veomni_commit": labels["org.opencontainers.image.revision"],
        "docker_commit": labels["ai.veomni.docker-revision"],
        "fla_commit": labels["ai.veomni.fla-revision"],
        "image_id": image["Id"],
        "repo_digests": image["RepoDigests"],
    }
    report["build"] = info
    if info["fla_commit"] != FLA_COMMIT:
        raise RuntimeError("Image uses another FLA commit")
    if info["veomni_commit"] != expected_docker_commit or info["docker_commit"] != expected_docker_commit:
        raise RuntimeError("Image was built from another branch commit")

    opp = Path(os.environ["ASCEND_OPP_PATH"])
    cann_info = (opp / "version.info").read_text()
    match = re.search(r"^\s*Version\s*=\s*[\"']?(9\.2\.1)(?:[\"']?\s*)$", cann_info, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Expected CANN 9.2.1 in {opp / 'version.info'}")
    report["cann_version"] = match.group(1)

    report["versions"] = {name: version(name) for name in EXPECTED_VERSIONS}
    report["versions"]["flash-linear-attention-npu-a5"] = version("flash-linear-attention-npu-a5")
    for name, expected in EXPECTED_VERSIONS.items():
        if report["versions"][name].split("+")[0] != expected:
            raise RuntimeError(f"Unexpected {name}: {report['versions'][name]}, expected {expected}")

    subprocess.run(["sha256sum", "--check", "/app/a5-ci-wheels.sha256"], check=True)
    report["wheel_checksums"] = "passed"
    # This upstream check runs in a fresh interpreter and must not auto-import torch_npu.
    subprocess.run(
        [sys.executable, "/app/check_packaged_wheel_api.py"],
        check=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )

    import fla_npu
    from fla_npu.ops import ascendc
    from fla_npu.ops.ascendc import _stable

    for name in ("chunk_gated_delta_rule_fwd", "chunk_gated_delta_rule_bwd"):
        if not callable(getattr(ascendc, name, None)):
            raise RuntimeError(f"Missing stable FLA API: {name}")
    if fla_npu.is_legacy_torch_ops_loaded():
        raise RuntimeError("Unexpected legacy FLA extension")
    _stable.load()
    if not _stable.available():
        raise RuntimeError("FLA Stable-ABI launcher did not load")

    import torch_npu  # noqa: F401
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    from triton._C.libtriton import ascend  # noqa: F401

    report["fla_stable_api"] = "passed"
    report["library_imports"] = "passed"

    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    report["pip_check"] = {"exit_code": result.returncode, "output": result.stdout + result.stderr}
    report["dependency_consistent"] = result.returncode == 0
    report["qualification"] = "packaging/imports only; dependency conflicts and NPU UT/ST remain separate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-inspect", type=Path, required=True)
    parser.add_argument("--expected-docker-commit", required=True)
    args = parser.parse_args()
    report = {"status": "failed", "python": sys.version, "architecture": platform.machine(), "npu_executed": False}
    try:
        check_image(report, args.image_inspect, args.expected_docker_commit)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
