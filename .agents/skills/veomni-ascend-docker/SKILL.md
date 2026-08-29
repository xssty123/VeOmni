---
name: veomni-ascend-docker
description: "Build and qualify VeOmni Ascend Docker candidate images through fork GitHub Actions, GHCR staging, and exact-digest smoke validation. Use for A2/A3 candidate builds, Ascend image environment checks, or freezing a validated image record. Quay promotion and real NPU UT/ST are out of scope unless explicitly requested."
---

# Ascend Docker Candidate Validation

## Outcome

Produce a candidate image once, store it in GHCR, validate the same immutable digest in a separate Actions job, and preserve enough evidence to reproduce the result.

Use the repository implementation instead of recreating it:

- Workflow: `.github/workflows/docker-build-ascend-candidate.yml`
- Smoke test: `docker/ascend/tests/image_smoke.py`

## Scope Gate

Before acting, identify whether the user asked to inspect, change, run, publish, or delete. Discussion and inspection do not authorize file edits, workflow runs, registry publication, default-branch changes, or deletion.

Apply the root `AGENTS.md` compatibility-evidence and allowed-diff rules. State the exact files and fields allowed to change before editing. Unless the user expands the scope, do not change:

- Ascend Dockerfiles or dependency versions
- `pyproject.toml` or `uv.lock`
- Existing A2/A3 Quay publishing workflows
- NPU UT/ST workflows
- Installation order, source-build commands, or workarounds

This skill's default mode is candidate validation only. It does not contain a Quay-promotion or registry-deletion runbook. When either is requested, use this skill only to freeze the validated digest and prepare an exact promotion or deletion plan; do not invent execution commands. Full UT/ST and real NPU forward/backward likewise require a separately scoped workflow.

## Existing Targets

The workflow builds one target per run:

| Target | Platform | Dockerfile | Runtime Python |
|---|---|---|---|
| `a2-amd64` | `linux/amd64` | `Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.x86` | `/app/.venv/bin/python` |
| `a2-arm64` | `linux/arm64` | `Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_910b.arm` | `python` |
| `a3-arm64` | `linux/arm64` | `Dockerfile.ascend_9.1.0_torch_npu2.10.0.post2_a3` | `python` |

Inspect the current workflow before relying on this table; version bumps may intentionally change the mapping.

## Procedure

### 1. Preflight

1. Inspect `git status`, remotes, the current branch, and the live fork default branch.
2. Preserve unrelated staged, unstaged, and untracked user files.
3. Identify the exact source commit and Dockerfile.
4. Resolve the fork's remote candidate-branch tip and require it to equal the intended full commit before dispatching. If the branch or commit exists only locally, do not silently switch refs; push only when the user's request authorizes publishing that branch.
5. If the request is run-only, do not edit the Dockerfile or workflow.

The workflow uses `workflow_dispatch`, so GitHub requires the workflow file to exist on the fork's default branch. If it is absent there, stop and request explicit authorization before modifying or pushing the default branch. When authorized, transplant only the workflow and smoke-test files; do not fast-forward or merge unrelated upstream commits merely to expose the dispatcher.

Repository visibility and GHCR package visibility are separate. Do not describe the candidate image as private merely because the repository is private, or vice versa. Verify package settings or authenticated-versus-anonymous pull behavior when privacy matters.

### 2. Run One Candidate

Trigger one architecture at a time so each run has an unambiguous result:

```bash
gh workflow run docker-build-ascend-candidate.yml \
  --repo <owner>/<fork> \
  --ref <candidate-branch> \
  -f target=a2-amd64
```

The workflow must keep these jobs separate:

1. `build-candidate` builds once and pushes a unique candidate tag to `ghcr.io/<owner>/veomni-ascend-candidate`.
2. `validate-candidate` starts on a fresh runner, authenticates to GHCR, and pulls `image@sha256:...`, never the tag alone.
3. `record-result` writes the final JSON record even when an earlier job fails.

Do not log in to Quay, publish manifests, or create formal release tags in this workflow.

### 3. Interpret Validation

The smoke test runs outside the Docker build and does not run `uv sync`. It exercises the environment already stored in the image:

- expected Python, torch, torch-npu, and triton-ascend versions
- `torch_npu`, `triton._C.libtriton.ascend`, and `fla_npu` loading
- registration of the `fla_npu` GDN `torch.ops.npu.*` operators
- image architecture

On a GitHub-hosted runner, `npu_available=false` is expected. The smoke test does not invoke `npu-smi`. A green smoke result proves packaging, imports, shared-library loading, and operator registration; it does not prove execution on NPU hardware.

Treat the candidate as passing this validation level only when the entire workflow concludes `success`. A successful build job alone is insufficient because pull-by-digest and smoke validation may still fail.

After the run, require the record's `commit` field, which is written from `GITHUB_SHA`, to equal the intended full commit. A matching branch name is not sufficient evidence because branch tips are mutable.

### 4. Freeze the Result

Record all of the following from the successful run:

- full source commit SHA
- Actions run URL
- target, platform, and Dockerfile
- full candidate tag
- immutable digest
- observed Python, torch, torch-npu, triton-ascend, and fla_npu versions
- the CANN version declared by the Dockerfile base image and the runtime `ASCEND_HOME_PATH` evidence; do not claim a stronger runtime-version check than the smoke log provides
- validation scope and its explicit NPU-runtime limitation

GHCR stores the image; Actions stores evidence in two places:

- `ascend-candidate-result.json` records identity and job-result fields: commit, run URL, target, platform, Dockerfile, image, tag, digest, build result, and validation result.
- The `Run image environment smoke test` log records the observed runtime versions, registered operators, NPU availability, and Ascend environment paths.

Capture both sources when freezing a baseline. Find the JSON under the run's `Artifacts` section. The artifact name begins with `ascend-candidate-` and contains `ascend-candidate-result.json`; do not claim that it contains the log-only runtime evidence.

If downloading all run artifacts fails on the automatically generated `.dockerbuild` record, list artifacts and download only the named result:

```bash
gh api repos/<owner>/<fork>/actions/runs/<run-id>/artifacts
gh run download <run-id> \
  --repo <owner>/<fork> \
  --name <ascend-candidate-artifact>
```

The digest, not the tag, is the frozen baseline. Tags remain mutable pointers even when their names include a commit and run ID.

### 5. Handle Failures

Classify the failure before changing anything:

- target resolution or runner allocation
- GHCR authentication or package permission
- base-image pull
- dependency installation
- triton-ascend loading
- fla_npu build or registration
- GHCR push
- digest pull
- architecture check
- smoke test
- result artifact upload

Make one narrow repair per attempt and preserve successful candidate digests. After three distinct repair attempts fail, stop, summarize the evidence, and ask the user before adding another workaround.

## Destructive and Expanded Operations

Deleting GHCR tags or versions is destructive. An initial request to clean old candidates authorizes inventory and planning, not the final delete call. List the exact package, tags, digests, package version IDs, and expected survivors, then obtain confirmation immediately before deletion. Never infer deletion targets from a shared digest. Because this v1 has no registry-deletion runbook, stop after the confirmed inventory unless the user supplies or explicitly requests a separate deletion procedure.

If the user later requests NPU execution, add a separate hardware-backed validation layer that consumes the already frozen digest. Do not retrofit `uv sync`-based UT/ST and call that image validation. If the user requests Quay release, stop after freezing the validated digest and exact requested destination; route to a separately reviewed promotion procedure that copies the validated content rather than rebuilding it.
