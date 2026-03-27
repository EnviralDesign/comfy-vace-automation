# comfy-vace-automation

Custom nodes for a leaner VACE clip-joining workflow in ComfyUI.

This repo replaces the old manifest-oriented automation pack with a smaller
set of nodes built around the current in-memory workflow:

- collect multiple clips
- iterate joins in-memory
- prepare a two-clip VACE seam
- apply lightweight seam blending
- expose loop iteration state for seed math

The design goal is to keep the workflow mostly native, with custom nodes only
where ComfyUI still needs a little orchestration help.

## Included nodes

- `VACE Clip Collector`
- `VACE Clip Loop Start`
- `VACE Clip Loop End`
- `VACE Seed Int`
- `VACE Join Prep`
- `VACE Crossfade Transition`
- `VACE Clip List (Up To 3)`

`VACE Clip List (Up To 3)` is retained as a small prototype/helper node. The
preferred front-end for real use is `VACE Clip Collector`.

## What this pack does

- Collects multiple `VIDEO` inputs and derives `IMAGE` clip batches plus shared FPS
- Loops across an ordered in-memory clip list without saving intermediate videos
- Carries the accumulated joined clip forward between iterations
- Prepares native VACE control frames and masks for a single seam
- Provides a standalone seed `INT` node with native Comfy seed-widget behavior

## What this pack does not do

- It does not bundle `WanVideoNAG`
- It does not bundle `ColorMatch`
- It does not bundle folder-manifest planning from the old workflow

If your workflow uses `WanVideoNAG` or `ColorMatch`, those still come from
external node packs such as KJNodes.

## Installation

Clone or copy this repo into your ComfyUI `custom_nodes` directory:

```text
ComfyUI/custom_nodes/comfy-vace-automation
```

Then restart ComfyUI.

## Dependencies

No extra pip packages are required beyond the normal ComfyUI runtime for the
core nodes in this repo.

Notes:

- `torch` is expected to come from the ComfyUI environment.
- The nodes use ComfyUI internals such as `nodes`, `comfy_execution`,
  and `comfy_api.latest`.
- Example workflows may still rely on external packs for quality extras such as
  `WanVideoNAG` and `ColorMatch`.

## Repo layout

- [`__init__.py`](C:/repos/comfy-vace-automation/__init__.py)
- [`collector_nodes.py`](C:/repos/comfy-vace-automation/collector_nodes.py)
- [`join_nodes.py`](C:/repos/comfy-vace-automation/join_nodes.py)

## Status

This repo now tracks the newer in-memory VACE join flow and supersedes the old
manifest/file-based automation pack that previously lived here.
