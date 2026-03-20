import json
import os
import time
import uuid

import av
import folder_paths
import numpy as np
import torch
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
    from comfy_execution.graph import ExecutionBlocker
except Exception:
    GraphBuilder = None
    ExecutionBlocker = None


def _make_run_token():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"run-{timestamp}-{uuid.uuid4().hex[:6]}"


def _first(value, default=None):
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _normalize_input_list(value):
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if value in (None, ""):
        return []
    return [str(value)]


def _manifest_filename():
    return "vace-run-manifest.json"


def _planned_output_path(output_dir, prefix):
    return os.path.join(output_dir, f"{prefix}_00001.mkv")


def _build_manifest(input_list, input_dir, project_name, debug):
    run_token = _make_run_token()
    run_dir_rel = f"{project_name}/vace-work/{run_token}" if project_name else f"vace-work/{run_token}"
    output_dir = folder_paths.get_output_directory()
    run_dir_abs = os.path.join(output_dir, run_dir_rel)
    manifest_path = os.path.join(run_dir_abs, _manifest_filename())

    os.makedirs(run_dir_abs, exist_ok=True)

    pairs = []
    final_sequence = []
    max_index = len(input_list) - 2

    for pair_index in range(max_index + 1):
        base_prefix = f"{run_dir_rel}/index{pair_index:03d}"
        video_1_name = input_list[pair_index]
        video_2_name = input_list[pair_index + 1]
        video_1_filename = os.path.join(input_dir, video_1_name) if input_dir else video_1_name
        video_2_filename = os.path.join(input_dir, video_2_name) if input_dir else video_2_name

        clip1_prefix = f"{base_prefix}_clip1"
        clip2_prefix = f"{base_prefix}_clip2"
        clip3_prefix = f"{base_prefix}_clip3"

        clip1_path = _planned_output_path(output_dir, clip1_prefix)
        clip2_path = _planned_output_path(output_dir, clip2_prefix)
        clip3_path = _planned_output_path(output_dir, clip3_prefix)

        is_last = pair_index == max_index

        final_sequence.append(clip1_path)
        final_sequence.append(clip2_path)
        if is_last:
            final_sequence.append(clip3_path)

        pairs.append(
            {
                "pair_index": pair_index,
                "video_1_filename": video_1_filename,
                "video_2_filename": video_2_filename,
                "clip1_prefix": clip1_prefix,
                "clip2_prefix": clip2_prefix,
                "clip3_prefix": clip3_prefix,
                "clip1_path": clip1_path,
                "clip2_path": clip2_path,
                "clip3_path": clip3_path,
                "is_first": pair_index == 0,
                "is_last": is_last,
            }
        )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_token": run_token,
        "input_dir": input_dir,
        "project_name": project_name,
        "run_dir": run_dir_rel,
        "run_dir_abs": run_dir_abs,
        "manifest_path": manifest_path,
        "input_list": input_list,
        "pair_count": len(pairs),
        "pairs": pairs,
        "final_sequence": final_sequence,
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if debug:
        print(f"[VACE Manifest] Created: {manifest_path}")
        print(f"[VACE Manifest] Run dir: {run_dir_rel}")
        print(f"[VACE Manifest] Pair count: {len(pairs)}")

    return manifest


def _load_manifest(manifest_path):
    manifest_path = manifest_path.strip().strip('"').strip("'")
    if not manifest_path:
        raise ValueError("Manifest path is empty")
    if not os.path.isfile(manifest_path):
        raise ValueError(f"Manifest file does not exist: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class VACEManifestLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": ("STRING", {"forceInput": True}),
                "input_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing input videos",
                }),
                "project_name": ("STRING", {
                    "default": "",
                    "tooltip": "Project name - workflow files will be created under ComfyUI/output/project_name.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log manifest planning and loop progress",
                }),
            },
            "optional": {
                "pair_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Internal loop state. Leave unconnected.",
                }),
                "manifest_path": ("STRING", {
                    "default": "",
                    "tooltip": "Internal loop state. Leave unconnected.",
                }),
            },
        }

    RETURN_TYPES = ("FLOW_CONTROL", "INT", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = (
        "flow",
        "pair_index",
        "manifest_path",
        "run_dir",
        "clip1_prefix",
        "clip2_prefix",
        "clip3_prefix",
        "video_1_filename",
        "video_2_filename",
        "is_first",
        "is_last",
    )
    FUNCTION = "loop_start"
    CATEGORY = "video/VACE"
    DESCRIPTION = """
    Creates a deterministic run manifest up front and emits per-pair context
    from that manifest for each loop iteration.
    """
    INPUT_IS_LIST = True

    def loop_start(self, **kwargs):
        input_list = _normalize_input_list(kwargs.get("input_list", []))
        input_dir = str(_first(kwargs.get("input_dir"), "") or "").strip()
        project_name = str(_first(kwargs.get("project_name"), "") or "").strip()
        debug = bool(_first(kwargs.get("debug"), False))
        pair_index = int(_first(kwargs.get("pair_index"), 0) or 0)
        manifest_path = str(_first(kwargs.get("manifest_path"), "") or "").strip()

        if len(input_list) < 2:
            raise ValueError(f"Need at least 2 videos to create transitions, found {len(input_list)}")

        manifest = _load_manifest(manifest_path) if manifest_path else _build_manifest(input_list, input_dir, project_name, debug)
        manifest_path = manifest["manifest_path"]

        if pair_index < 0 or pair_index >= manifest["pair_count"]:
            raise ValueError(f"Pair index {pair_index} out of range (valid: 0-{manifest['pair_count'] - 1})")

        pair = manifest["pairs"][pair_index]

        if debug:
            print("\n[VACE Manifest Loop Start] === Start ===")
            print(f"[VACE Manifest Loop Start] Pair index: {pair_index} (videos {pair_index + 1}-{pair_index + 2} of {len(manifest['input_list'])})")
            print(f"[VACE Manifest Loop Start] Manifest: {manifest_path}")
            print(f"[VACE Manifest Loop Start] {'[FIRST]' if pair['is_first'] else ''} {'[LAST]' if pair['is_last'] else ''}")
            print(f"[VACE Manifest Loop Start] Video 1: {os.path.basename(pair['video_1_filename'])}")
            print(f"[VACE Manifest Loop Start] Video 2: {os.path.basename(pair['video_2_filename'])}")
            print(f"[VACE Manifest Loop Start] Clip1 prefix: {pair['clip1_prefix']}")
            print(f"[VACE Manifest Loop Start] Clip2 prefix: {pair['clip2_prefix']}")
            print(f"[VACE Manifest Loop Start] Clip3 prefix: {pair['clip3_prefix']}")
            print(f"[VACE Manifest Loop Start] === End ===")

        return (
            "stub",
            pair_index,
            manifest_path,
            manifest["run_dir"],
            pair["clip1_prefix"],
            pair["clip2_prefix"],
            pair["clip3_prefix"],
            pair["video_1_filename"],
            pair["video_2_filename"],
            pair["is_first"],
            pair["is_last"],
        )


class VACEManifestLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "pair_index": ("INT", {"forceInput": True}),
                "manifest_path": ("STRING", {"forceInput": True}),
                "is_last": ("BOOLEAN", {"forceInput": True}),
                "clip2_saved_filename": ("STRING", {"forceInput": True}),
                "fps": ("FLOAT", {"forceInput": True}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("manifest_path", "fps")
    FUNCTION = "loop_end"
    CATEGORY = "video/VACE"

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return

        for _, value in node_info["inputs"].items():
            if is_link(value):
                parent_id = value[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]
                if class_type != "VACEManifestLoopEnd":
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)
                upstream[parent_id].append(node_id)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id, output_link in output_nodes.items():
                node_id = output_link[0]
                if node_id in parent_ids and display_id == node_id and output_id not in upstream[parent_id]:
                    if "." in parent_id:
                        parts = parent_id.split(".")
                        parts[-1] = output_id
                        upstream[parent_id].append(".".join(parts))
                    else:
                        upstream[parent_id].append(output_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def loop_end(self, flow, pair_index, manifest_path, is_last, clip2_saved_filename, fps, dynprompt=None, unique_id=None):
        if not clip2_saved_filename:
            return (ExecutionBlocker(None), ExecutionBlocker(None))

        if is_last:
            return (manifest_path, fps)

        if GraphBuilder is None:
            raise RuntimeError("GraphBuilder is unavailable; manifest VACE looping cannot execute")

        upstream = {}
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))

        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for node_id, node in prompts.items():
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            class_def = ALL_NODE_CLASS_MAPPINGS.get(class_type)
            if class_def is None:
                continue
            if hasattr(class_def, "OUTPUT_NODE") and class_def.OUTPUT_NODE is True:
                for _, value in node["inputs"].items():
                    if is_link(value):
                        output_nodes[node_id] = value

        graph = GraphBuilder()
        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)

        contained = {}
        open_node = flow[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for key, value in original_node["inputs"].items():
                if is_link(value) and value[0] in contained:
                    parent = graph.lookup_node(value[0])
                    node.set_input(key, parent.out(value[1]))
                else:
                    node.set_input(key, value)

        new_open = graph.lookup_node(open_node)
        new_open.set_input("pair_index", pair_index + 1)
        new_open.set_input("manifest_path", manifest_path)

        my_clone = graph.lookup_node("Recurse")
        return {
            "result": (my_clone.out(0), my_clone.out(1)),
            "expand": graph.finalize(),
        }


class VACEManifestLoadOrderedClips:
    VIDEO_EXTENSIONS = ["webm", "mp4", "mkv", "gif", "mov"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_path": ("STRING", {"default": ""}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load_videos"
    CATEGORY = "video/utility"
    DESCRIPTION = """
    Load clips in the exact order declared by a VACE run manifest.
    This avoids folder scanning and filename sort assumptions.
    """

    def load_videos(self, manifest_path, debug, meta_batch=None, unique_id=None):
        manifest = _load_manifest(manifest_path)
        video_files = self._get_video_files_from_manifest(manifest)

        if not video_files:
            raise ValueError(f"No planned output clips found in manifest: {manifest_path}")

        if meta_batch is None:
            return self._load_all(video_files, manifest["run_dir"], debug)
        return self._load_batched(video_files, manifest["run_dir"], debug, meta_batch, unique_id)

    def _get_video_files_from_manifest(self, manifest):
        video_files = []
        for path in manifest.get("final_sequence", []):
            if not os.path.isfile(path):
                raise ValueError(f"Planned clip is missing: {path}")
            ext = os.path.splitext(path)[1].lstrip(".").lower()
            if ext not in self.VIDEO_EXTENSIONS:
                raise ValueError(f"Unsupported planned clip extension: {path}")
            video_files.append(path)
        return video_files

    def _load_all(self, video_files, label, debug):
        if debug:
            print(f"[Load Videos Manifest] Loading {len(video_files)} videos from manifest for {label}")

        all_frames = []
        expected_shape = None

        for idx, video_path in enumerate(video_files):
            if debug:
                print(f"[Load Videos Manifest] [{idx+1}/{len(video_files)}]: {os.path.basename(video_path)}", end=" ... ")

            frames = self._load_video_frames(video_path)
            expected_shape = self._check_resolution(frames, expected_shape, video_path)
            all_frames.append(frames)

            if debug:
                print(f"{frames.shape[0]} frames")

        if debug:
            print(f"[Load Videos Manifest] Concatenating {len(video_files)} videos...")
        output = torch.cat(all_frames, dim=0)

        if debug:
            print("[Load Videos Manifest] Done")
        return (output,)

    def _load_batched(self, video_files, label, debug, meta_batch, unique_id):
        if unique_id not in meta_batch.inputs:
            total_frames = self._count_total_frames(video_files)
            meta_batch.total_frames = min(meta_batch.total_frames, total_frames)
            if debug:
                print(f"[Load Videos Manifest] Batched: Starting generator for {len(video_files)} videos ({total_frames} frames) in {label}")
            meta_batch.inputs[unique_id] = self._frame_generator(video_files, debug)

        generator = meta_batch.inputs[unique_id]
        frames_per_batch = meta_batch.frames_per_batch

        batch_frames = []
        expected_shape = None
        frames_collected = 0

        while frames_collected < frames_per_batch:
            try:
                frame_tensor, video_path = next(generator)
            except StopIteration:
                if debug:
                    print("[Load Videos Manifest] Batched: Generator exhausted, cleaning up")
                meta_batch.inputs.pop(unique_id)
                meta_batch.has_closed_inputs = True
                break

            expected_shape = self._check_resolution(frame_tensor.unsqueeze(0), expected_shape, video_path)
            batch_frames.append(frame_tensor)
            frames_collected += 1

        if not batch_frames:
            raise RuntimeError("Manifest loader produced no frames")

        output = torch.stack(batch_frames, dim=0)

        if debug:
            print(f"[Load Videos Manifest] Batched: Yielding {output.shape[0]} frames  shape={tuple(output.shape)}")

        return (output,)

    def _load_video_frames(self, video_path):
        container = av.open(video_path)
        try:
            if len(container.streams.video) == 0:
                raise ValueError(f"No video stream found in: {video_path}")

            frames = []
            for frame in container.decode(video=0):
                rgb = frame.to_ndarray(format="rgb24")
                frame_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
                frames.append(frame_tensor)

            if not frames:
                raise RuntimeError(f"No frames extracted from {video_path}")

            return torch.stack(frames, dim=0)
        finally:
            container.close()

    def _check_resolution(self, frames, expected_shape, video_path):
        if expected_shape is None:
            return frames.shape[1:3]
        if frames.shape[1:3] != expected_shape:
            raise ValueError(
                f"\nResolution mismatch\n"
                f"  Expected: {expected_shape[1]}x{expected_shape[0]} (from first clip)\n"
                f"  Got: {frames.shape[2]}x{frames.shape[1]} in {os.path.basename(video_path)}"
            )
        return expected_shape

    def _count_total_frames(self, video_files):
        total = 0
        for video_path in video_files:
            container = None
            try:
                container = av.open(video_path)
                if len(container.streams.video) > 0:
                    count = container.streams.video[0].frames
                    total += count if count > 0 else sum(1 for _ in container.decode(video=0))
            finally:
                if container is not None:
                    container.close()
        return total

    def _frame_generator(self, video_files, debug):
        for idx, video_path in enumerate(video_files):
            if debug:
                print(f"[Load Videos Manifest] Batched: Opening [{idx+1}/{len(video_files)}]: {os.path.basename(video_path)}")

            container = av.open(video_path)
            try:
                if len(container.streams.video) == 0:
                    raise ValueError(f"No video stream found in: {video_path}")
                for frame in container.decode(video=0):
                    rgb = frame.to_ndarray(format="rgb24")
                    yield torch.from_numpy(rgb.astype(np.float32) / 255.0), video_path
            finally:
                container.close()


NODE_CLASS_MAPPINGS = {
    "VACEManifestLoopStart": VACEManifestLoopStart,
    "VACEManifestLoopEnd": VACEManifestLoopEnd,
    "VACEManifestLoadOrderedClips": VACEManifestLoadOrderedClips,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VACEManifestLoopStart": "🪐 VACE Manifest Loop Start",
    "VACEManifestLoopEnd": "🪐 VACE Manifest Loop End",
    "VACEManifestLoadOrderedClips": "🪐 Load Videos From VACE Manifest",
}
