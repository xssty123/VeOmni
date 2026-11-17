"""Check the retained recipes' installed packages without NPU computation."""

import argparse
import importlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


PROFILES = {
    "a2-amd64": ("x86_64", "a2", "3.2.0", "3.2.2"),
    "a2-arm64": ("aarch64", "a2", "3.5.0", "3.2.2"),
    "a3-arm64": ("aarch64", "a3", "3.5.0", "3.2.2"),
    "a5-arm64": ("aarch64", "a5", "3.6.0", "3.6.0"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=PROFILES)
    args = parser.parse_args()
    machine, product, triton, ascend = PROFILES[args.target]
    if platform.machine() != machine or sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Expected {machine}, Python 3.12")
    expected = {
        "torch": "2.10.0",
        "torch-npu": "2.10.0.post7",
        "triton": triton,
        "triton-ascend": ascend,
        "transformers": "5.16.1",
    }
    actual = {name: version(name) for name in expected}
    for name, pin in expected.items():
        if actual[name].split("+")[0] != pin:
            raise RuntimeError(f"Expected {name} {pin}, found {actual[name]}")
    for module in ("torch", "torch_npu", "fla_npu", "torchcodec.decoders"):
        importlib.import_module(module)
    libtriton = importlib.import_module("triton._C.libtriton")
    if not hasattr(libtriton, "ascend"):
        raise RuntimeError("Ascend Triton backend is missing")
    from fla_npu.ops import ascendc

    for name in ("chunk_gated_delta_rule_fwd", "chunk_gated_delta_rule_bwd"):
        if not callable(getattr(ascendc, name, None)):
            raise RuntimeError(f"Missing FLA stable API: {name}")
    subprocess.run(["sha256sum", "-c", f"/app/{product}-ci-wheels.sha256"], check=True)
    subprocess.run([sys.executable, "/app/check_packaged_wheel_api.py"], check=True)
    actual[f"flash-linear-attention-npu-{product}"] = version(f"flash-linear-attention-npu-{product}")
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
    print(
        json.dumps(
            {
                "target": args.target,
                "versions": actual,
                "pip_check": {"exit_code": pip_check.returncode, "output": pip_check.stdout + pip_check.stderr},
                "npu_ut_st": "not_run",
                "scope": "packaging and imports only",
                "requirements_record": str(Path(f"/app/{product}-image-requirements.txt")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
