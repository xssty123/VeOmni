"""Check a pinned A5 daily-wheel image without installing dependencies or running NPU kernels."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from check_daily_build_contract import check_installed


def check_image(report, args, profile):
    if platform.machine() != profile["architecture"] or list(sys.version_info[:2]) != profile["python"]:
        raise RuntimeError("Image Python/architecture differs from the selected profile")
    image = json.loads(args.image_inspect.read_text())[0]
    labels = image["Config"]["Labels"]
    expected_labels = {
        "org.opencontainers.image.revision": args.expected_docker_commit,
        "ai.veomni.docker-revision": args.expected_docker_commit,
        "ai.veomni.target": profile["target"],
        "ai.veomni.fla-version": profile["fla"]["version"],
        "ai.veomni.fla-wheel-sha256": profile["fla"]["sha256"],
    }
    for name, expected in expected_labels.items():
        if labels.get(name) != expected:
            raise RuntimeError(f"Image label mismatch: {name}")
    if image["Architecture"] != "arm64" or image["Os"] != "linux":
        raise RuntimeError("Registry image is not Linux ARM64")
    report["build"] = {
        "docker_commit": args.expected_docker_commit,
        "image_id": image["Id"],
        "repo_digests": image["RepoDigests"],
        "labels": labels,
    }
    opp = Path(os.environ["ASCEND_OPP_PATH"])
    cann_info = (opp / "version.info").read_text()
    expected_cann = re.escape(profile["cann"])
    if not re.search(rf"^\s*Version\s*=\s*[\"']?{expected_cann}[\"']?\s*$", cann_info, re.MULTILINE):
        raise RuntimeError(f"Expected CANN {profile['cann']} in {opp / 'version.info'}")
    report["cann_version"] = profile["cann"]
    report["versions"] = check_installed(profile)
    report["artifact_checksums"] = {}
    for name in ("fla", "triton_ascend", "fla_checker"):
        artifact = profile[name]
        filename = artifact.get("filename", "check_packaged_wheel_api.py")
        path = Path("/app") / filename
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != artifact["sha256"]:
            raise RuntimeError(f"Retained {name} artifact differs from the reviewed SHA256")
        report["artifact_checksums"][name] = digest
    subprocess.run(["sha256sum", "--check", "/app/a5-ci-wheels.sha256"], check=True)
    # The pinned publisher script validates all 38 actual Stable registrations,
    # OPP payloads and absence of legacy/fallback paths, without executing kernels.
    subprocess.run(
        [sys.executable, "/app/check_packaged_wheel_api.py"],
        check=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )
    import torch_npu  # noqa: F401
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    from triton._C.libtriton import ascend  # noqa: F401

    report["packaged_fla_api"] = "passed"
    report["library_imports"] = "passed"
    report["fla_source_provenance"] = profile["fla"]["source_provenance"]
    report["qualification"] = "packaging/imports only; dependency consistency and NPU UT/ST are separate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-inspect", type=Path, required=True)
    parser.add_argument("--expected-docker-commit", required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    report = {
        "status": "failed",
        "profile": profile["name"],
        "python": sys.version,
        "architecture": platform.machine(),
        "npu_executed": False,
    }
    try:
        check_image(report, args, profile)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        report["pip_check"] = {"exit_code": result.returncode, "output": result.stdout + result.stderr}
        report["dependency_consistent"] = result.returncode == 0
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
