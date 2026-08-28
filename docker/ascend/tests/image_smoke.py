from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import sys


EXPECTED_PYTHON = (3, 12)
EXPECTED_TORCH = "2.10.0"
EXPECTED_TORCH_NPU = "2.10.0.post2"
EXPECTED_TRITON_ASCEND = "3.2.1"
REQUIRED_FLA_NPU_OPS = (
    "npu_chunk_bwd_dqkwg",
    "npu_chunk_bwd_dv_local",
    "npu_chunk_fwd_o",
    "npu_chunk_gated_delta_rule_bwd_dhu",
    "npu_chunk_gated_delta_rule_fwd_h",
    "npu_prepare_wy_repr_bwd_da",
    "npu_prepare_wy_repr_bwd_full",
    "npu_recompute_w_u_fwd",
)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-found"


def main() -> None:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Expected Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}, got {platform.python_version()}"
        )

    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    importlib.import_module("fla_npu")
    libtriton = importlib.import_module("triton._C.libtriton")

    if not hasattr(libtriton, "ascend"):
        raise RuntimeError("triton-ascend did not expose triton._C.libtriton.ascend")

    torch_version = str(torch.__version__)
    torch_npu_version = _distribution_version("torch-npu")
    triton_ascend_version = _distribution_version("triton-ascend")

    if torch_version.split("+", maxsplit=1)[0] != EXPECTED_TORCH:
        raise RuntimeError(f"Expected torch {EXPECTED_TORCH}, got {torch_version}")
    if torch_npu_version != EXPECTED_TORCH_NPU:
        raise RuntimeError(f"Expected torch-npu {EXPECTED_TORCH_NPU}, got {torch_npu_version}")
    if triton_ascend_version != EXPECTED_TRITON_ASCEND:
        raise RuntimeError(f"Expected triton-ascend {EXPECTED_TRITON_ASCEND}, got {triton_ascend_version}")

    missing_ops = [name for name in REQUIRED_FLA_NPU_OPS if not hasattr(torch.ops.npu, name)]
    if missing_ops:
        raise RuntimeError(f"fla_npu did not register required torch.ops.npu operators: {missing_ops}")

    try:
        npu_available: bool | str = bool(torch.npu.is_available())
    except Exception as error:  # noqa: BLE001
        npu_available = f"unavailable without an NPU runner: {error}"

    result = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "torch": torch_version,
        "torch_npu": torch_npu_version,
        "triton_ascend": triton_ascend_version,
        "fla_npu": _distribution_version("fla-npu"),
        "registered_fla_npu_ops": list(REQUIRED_FLA_NPU_OPS),
        "npu_available": npu_available,
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH"),
        "ascend_opp_path": os.environ.get("ASCEND_OPP_PATH"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
