# Docker branch consolidation contract

This is preservation of existing image recipes and registry customization, not a dependency refresh or a new successful image baseline.

The upstream base and all source tips are fixed in `branch-consolidation-20261117.json`. The four CANN 9.2.1 Dockerfiles and three corresponding build guides are copied byte for byte from the existing `c58f0f4b` worktree, including its six pending files. Their hashes are recorded before editing workflow files. A2 x86 retains its source-build route; the pending ARM prebuilt FLA wheel changes are retained exactly. No package pin, installation command, workaround or recipe assertion may change during this consolidation.

Allowed changes are: copy those existing recipes/guides, restore the GHCR candidate workflow for all four platform targets, restore the three direct Quay build workflows with product-based tags, add the shared tag helper and packaging smoke, preserve historical publishing YAML outside `.github/workflows`, and document these entrypoints. All test and model code stays at the fixed upstream base.

No successful build is claimed for the copied pending recipes. The historical recorded A5 image has digest `sha256:369b251a90d189051d0f95038345440869f105d6753395958223a11ed89aee95` and source label `273d31cbf9eebd05913c38ed15388447a0f094f2`; it is a separate packaging/import reference, not validation of these recipes. No image build, dependency installation, NPU CI, Quay publication or upstream PR is part of the consolidation.
