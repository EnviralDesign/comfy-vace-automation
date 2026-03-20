# comfy-vace-automation

Custom nodes for ComfyUI workflows that automate multi-clip VACE runs.

This repo contains a small set of workflow-control and manifest-planning nodes
used to make repeated VACE transition runs more deterministic and easier to
chain together.

## Included nodes

- `WanVACEBatchContextAuto`
- `VACEPairLoopStart`
- `VACEPairLoopEnd`
- `VACEManifestLoopStart`
- `VACEManifestLoopEnd`
- `VACEManifestLoadOrderedClips`
- `VACEReadyOutputPath`

These nodes appear under the `video/VACE` category in ComfyUI, except
`VACEManifestLoadOrderedClips`, which appears under `video/utility`.

## What it does

- Creates stable per-run work directories and prefixes for VACE output clips
- Supports looping over adjacent input-video pairs inside a single prompt run
- Builds a manifest that declares expected clip paths up front
- Loads generated clips back in manifest order instead of relying on folder scans
- Releases a final output path only when the workflow has produced a ready signal

## Installation

Clone or copy this repo into your ComfyUI `custom_nodes` directory:

```text
ComfyUI/custom_nodes/comfyui-vace-automation
```

Install the Python dependency:

```bash
pip install -r requirements.txt
```

Then restart ComfyUI.

## Dependencies

- ComfyUI
- `av`
- `numpy`

Notes:

- `torch` is expected to come from the ComfyUI environment.
- `folder_paths`, `nodes`, and `comfy_execution` are ComfyUI internals and are
  not standalone pip dependencies.
- `VACEManifestLoadOrderedClips` can accept an optional `VHS_BatchManager`
  input if VideoHelperSuite is present, but the node does not require VHS for
  its basic non-batched path.

## Repo layout

- [`__init__.py`](C:/repos/comfy-vace-automation/__init__.py)
- [`batch_context_auto.py`](C:/repos/comfy-vace-automation/batch_context_auto.py)
- [`final_join_gate.py`](C:/repos/comfy-vace-automation/final_join_gate.py)
- [`manifest_run.py`](C:/repos/comfy-vace-automation/manifest_run.py)
- [`single_run_loop.py`](C:/repos/comfy-vace-automation/single_run_loop.py)

## Status

This is a formalized copy of a working in-place custom node that was originally
developed directly inside a local ComfyUI install. The node logic has been kept
intact; cleanup here is limited to packaging the source into a clean repo.
