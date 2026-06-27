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
- `VACE Join Assemble`
- `VACE Final Loop Prep`
- `VACE Final Loop Assemble`
- `VACE Load Video From Path`
- `VACE Two Video Bridge Prep`
- `VACE Two Video Bridge Assemble`
- `VACE Clip List (Up To 3)`

`VACE Clip List (Up To 3)` is retained as a small prototype/helper node. The
preferred front-end for real use is `VACE Clip Collector`.

## What this pack does

- Collects multiple `VIDEO` inputs and derives `IMAGE` clip batches plus shared FPS
- Loops across an ordered in-memory clip list without saving intermediate videos
- Optionally adds one final tail-to-head loop pass after the last normal seam
- Carries the accumulated joined clip forward between iterations
- Prepares native VACE control frames and masks for a single seam
- Assembles normal seams and final loop seams without duplicating or extending the first clip body
- Prepares and assembles a final tail-to-head VACE loop pass for one clip
- Loads native Comfy `VIDEO` objects from absolute or project-relative paths
- Prepares and assembles direct left/right video bridges for NLE-style workflows
- Provides a standalone seed `INT` node with native Comfy seed-widget behavior

## Loading videos from project paths

Use `VACE Load Video From Path` when the video file lives outside ComfyUI's
input directory.

Inputs:

- `path`: absolute video path, or a relative video path
- `base_directory`: optional base directory for relative project paths

If `path` is relative, the node checks `base_directory` first when provided,
then ComfyUI's input directory, then the Comfy process working directory. The
output is a native Comfy `VIDEO`, so it can connect directly to nodes such as
`VACE Two Video Bridge Prep`.

## Direct two-video bridge

For editor/NLE use, prefer the `VACE Two Video Bridge Prep` and
`VACE Two Video Bridge Assemble` nodes.

`VACE Two Video Bridge Prep` takes two native `VIDEO` inputs:

- `left_video`: clip before the seam
- `right_video`: clip after the seam
- `left_replace_frames`: frames replaced from the end of the left video
- `right_replace_frames`: frames replaced from the start of the right video
- `edge_blend_frames`: outer bridge frames used as source anchors and final blend edges
- `fps`: optional int/float force-input override; leave disconnected to derive and validate FPS from both videos
- `resize_strategy`: `passthrough` keeps the current matching-input-size behavior; `explicit_resize` stretches both videos to `width` and `height`
- `width` / `height`: optional int force-input targets used only when `resize_strategy` is `explicit_resize`
- `color_reference_frame`: one-frame output for color matching; uses the last non-gray left control anchor, falling back to the left span/input frame

The visible bridge span is:

```text
left_replace_frames + right_replace_frames
```

The node hides Wan's `4n + 1` generation length requirement by inserting masked
padding at the seam. For example, a 32-frame visible bridge becomes a 33-frame
Wan control window, and the assemble node trims it back to 32 frames.

`VACE Two Video Bridge Assemble` consumes the generated bridge frames after VACE
decode and returns both:

- bridge-only images/video
- full joined images/video

Its `edge_blend_easing` setting controls the same edge-blend curve that the old
crossfade node used when blending generated bridge edges back into source frames.

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

## Donations & Support

If this saves you time, you can support the work here:

- [Patreon](https://www.patreon.com/EnviralDesign)
- [GitHub Sponsors](https://github.com/sponsors/EnviralDesign)
- [PayPal](https://www.paypal.com/donate?hosted_button_id=RP8EJAHSDTZ86)
