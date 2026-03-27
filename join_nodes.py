import torch
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS

try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
    from comfy_execution.graph import ExecutionBlocker
except Exception:
    GraphBuilder = None
    ExecutionBlocker = None


def _validate_video_tensor(name, video):
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if video.ndim != 4:
        raise ValueError(f"{name} must be IMAGE batch data shaped [frames, height, width, channels]")
    if video.shape[-1] < 3:
        raise ValueError(f"{name} must have at least 3 channels")


def _first(value, default=None):
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _normalize_clip_list(clips):
    if clips is None:
        return []
    if not isinstance(clips, list):
        clips = [clips]
    return [clip for clip in clips if clip is not None]


def _context_slice(video, context_frames, replace_frames, from_start):
    trim = context_frames + replace_frames
    if replace_frames > 0:
        if from_start:
            return video[replace_frames:trim]
        return video[-trim:-replace_frames]
    if from_start:
        return video[:context_frames]
    return video[-context_frames:]


def _trim_outer(video, trim, from_start):
    if trim <= 0:
        return video
    if from_start:
        return video[trim:]
    return video[:-trim]


def _apply_easing(values, easing):
    if easing == "linear":
        return values
    if easing == "ease_in":
        return values * values
    if easing == "ease_out":
        return 1.0 - (1.0 - values) * (1.0 - values)
    if easing == "ease_in_out":
        lower = 2.0 * values * values
        upper = 1.0 - torch.pow(-2.0 * values + 2.0, 2) / 2.0
        return torch.where(values < 0.5, lower, upper)
    raise ValueError(f"Unsupported easing mode: {easing}")


class VACEClipList3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_1": ("IMAGE",),
                "clip_2": ("IMAGE",),
                "fps_1": ("FLOAT", {"forceInput": True}),
                "fps_2": ("FLOAT", {"forceInput": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "clip_3": ("IMAGE",),
                "fps_3": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("clips", "fps", "clip_count")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "build"
    CATEGORY = "video/VACE"
    DESCRIPTION = "Build a small ordered in-memory clip list for VACE looping and validate shared FPS."

    def build(self, clip_1, clip_2, fps_1, fps_2, debug, clip_3=None, fps_3=None):
        clips = [clip_1, clip_2]
        fps_values = [float(fps_1), float(fps_2)]

        if clip_3 is not None:
            if fps_3 is None:
                raise ValueError("fps_3 is required when clip_3 is connected")
            clips.append(clip_3)
            fps_values.append(float(fps_3))

        if len(clips) < 2:
            raise ValueError("Need at least 2 clips")

        for idx, clip in enumerate(clips, start=1):
            _validate_video_tensor(f"clip_{idx}", clip)

        base_shape = clips[0].shape[1:]
        for idx, clip in enumerate(clips[1:], start=2):
            if clip.shape[1:] != base_shape:
                raise ValueError(
                    f"All clips must share the same resolution/channels, "
                    f"clip_1={tuple(base_shape)} vs clip_{idx}={tuple(clip.shape[1:])}"
                )

        output_fps = fps_values[0]
        for idx, value in enumerate(fps_values[1:], start=2):
            if abs(value - output_fps) > 1e-6:
                raise ValueError(
                    f"All clips must share the same FPS, fps_1={output_fps} vs fps_{idx}={value}"
                )

        if debug:
            print("\n[VACEClipList3] === Start ===")
            print(f"[VACEClipList3] clip_count: {len(clips)}")
            print(f"[VACEClipList3] fps: {output_fps}")
            for idx, clip in enumerate(clips, start=1):
                print(f"[VACEClipList3] clip_{idx}: frames={clip.shape[0]} shape={tuple(clip.shape[1:])}")
            print("[VACEClipList3] === End ===")

        return (clips, output_fps, len(clips))


class VACEClipLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clips": ("IMAGE", {"forceInput": True}),
                "fps": ("FLOAT", {"forceInput": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "next_index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Internal loop state. Leave unconnected.",
                }),
                "accumulator_clip": ("IMAGE", {
                    "tooltip": "Internal loop state. Leave unconnected.",
                }),
            },
        }

    RETURN_TYPES = ("FLOW_CONTROL", "INT", "INT", "FLOAT", "IMAGE", "IMAGE", "BOOLEAN", "INT")
    RETURN_NAMES = ("flow", "next_index", "clip_count", "fps", "left_clip", "right_clip", "is_last", "iteration_index")
    FUNCTION = "loop_start"
    CATEGORY = "video/VACE"
    INPUT_IS_LIST = True
    DESCRIPTION = "Emit the current in-memory VACE clip pair while carrying an accumulated joined clip between iterations."

    def loop_start(self, clips, fps, debug, next_index=None, accumulator_clip=None):
        clip_list = _normalize_clip_list(clips)
        if len(clip_list) < 2:
            raise ValueError(f"Need at least 2 clips to loop, found {len(clip_list)}")

        fps_value = float(_first(fps, 0.0) or 0.0)
        if fps_value <= 0:
            raise ValueError(f"FPS must be > 0, got {fps_value}")

        next_index_value = int(_first(next_index, 1) or 1)
        if next_index_value < 1 or next_index_value >= len(clip_list):
            raise ValueError(
                f"next_index {next_index_value} out of range "
                f"(valid: 1-{len(clip_list) - 1})"
            )

        accumulator = _first(accumulator_clip, None)
        if accumulator is not None:
            _validate_video_tensor("accumulator_clip", accumulator)
            left_clip = accumulator
        else:
            left_clip = clip_list[0]

        right_clip = clip_list[next_index_value]
        _validate_video_tensor("left_clip", left_clip)
        _validate_video_tensor("right_clip", right_clip)

        if left_clip.shape[1:] != right_clip.shape[1:]:
            raise ValueError(
                f"Loop pair resolution mismatch: left_clip={tuple(left_clip.shape[1:3])}, "
                f"right_clip={tuple(right_clip.shape[1:3])}"
            )

        is_last = next_index_value == len(clip_list) - 1
        iteration_index = next_index_value - 1
        debug_value = bool(_first(debug, False))
        if debug_value:
            print("\n[VACEClipLoopStart] === Start ===")
            print(f"[VACEClipLoopStart] clip_count: {len(clip_list)}")
            print(f"[VACEClipLoopStart] next_index: {next_index_value}")
            print(f"[VACEClipLoopStart] iteration_index: {iteration_index}")
            print(f"[VACEClipLoopStart] fps: {fps_value}")
            print(f"[VACEClipLoopStart] accumulator: {'yes' if accumulator is not None else 'no'}")
            print(f"[VACEClipLoopStart] left_clip frames: {left_clip.shape[0]}")
            print(f"[VACEClipLoopStart] right_clip frames: {right_clip.shape[0]}")
            print(f"[VACEClipLoopStart] is_last: {is_last}")
            print("[VACEClipLoopStart] === End ===")

        return ("stub", next_index_value, len(clip_list), fps_value, left_clip, right_clip, is_last, iteration_index)


class VACEClipLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "next_index": ("INT", {"forceInput": True}),
                "is_last": ("BOOLEAN", {"forceInput": True}),
                "joined_clip": ("IMAGE", {"forceInput": True}),
                "fps": ("FLOAT", {"forceInput": True}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "loop_end"
    CATEGORY = "video/VACE"
    DESCRIPTION = "Close the in-memory clip loop and recurse until all clips are joined into one IMAGE batch."

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
                if class_type != "VACEClipLoopEnd":
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

    def loop_end(self, flow, next_index, is_last, joined_clip, fps, dynprompt=None, unique_id=None):
        _validate_video_tensor("joined_clip", joined_clip)

        if is_last:
            return (joined_clip, fps)

        if GraphBuilder is None:
            raise RuntimeError("GraphBuilder is unavailable; in-memory VACE looping cannot execute")

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
        new_open.set_input("next_index", int(next_index) + 1)
        new_open.set_input("accumulator_clip", joined_clip)

        my_clone = graph.lookup_node("Recurse")
        return {
            "result": (my_clone.out(0), my_clone.out(1)),
            "expand": graph.finalize(),
        }


class VACESeedInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Standalone seed integer. Use the widget mode for fixed, increment, decrement, or randomize.",
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "passthrough"
    CATEGORY = "video/VACE"
    DESCRIPTION = "Expose a standalone seed INT with native Comfy control-after-generate behavior so it can be combined with loop iteration math."

    def passthrough(self, seed):
        return (int(seed),)


class VACEJoinPrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_1": ("IMAGE",),
                "video_2": ("IMAGE",),
                "context_frames": ("INT", {"default": 8, "min": 1, "max": 4096}),
                "replace_frames": ("INT", {"default": 8, "min": 0, "max": 4096}),
                "new_frames": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = (
        "control_video",
        "control_mask",
        "width",
        "height",
        "length",
        "start_images",
        "end_images",
        "context_frames",
        "replace_frames",
        "new_frames",
    )
    FUNCTION = "prepare"
    CATEGORY = "video/VACE"
    DESCRIPTION = "Prepare a two-clip VACE seam from explicit video inputs without any loop or manifest logic."

    def prepare(self, video_1, video_2, context_frames, replace_frames, new_frames, debug):
        _validate_video_tensor("video_1", video_1)
        _validate_video_tensor("video_2", video_2)

        if video_1.shape[1:] != video_2.shape[1:]:
            raise ValueError(
                f"Video resolution mismatch: video_1={tuple(video_1.shape[1:3])}, "
                f"video_2={tuple(video_2.shape[1:3])}"
            )

        height = int(video_1.shape[1])
        width = int(video_1.shape[2])
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError(f"Video dimensions must be divisible by 16, got {width}x{height}")

        trim = context_frames + replace_frames
        required_frames = trim if replace_frames > 0 else context_frames
        if video_1.shape[0] < required_frames:
            raise ValueError(f"video_1 needs at least {required_frames} frames, got {video_1.shape[0]}")
        if video_2.shape[0] < required_frames:
            raise ValueError(f"video_2 needs at least {required_frames} frames, got {video_2.shape[0]}")

        v1_context = _context_slice(video_1, context_frames, replace_frames, from_start=False)
        v2_context = _context_slice(video_2, context_frames, replace_frames, from_start=True)

        filler_count = (replace_frames * 2) + new_frames + 1
        filler = torch.full(
            (filler_count, height, width, video_1.shape[3]),
            0.5,
            dtype=video_1.dtype,
            device=video_1.device,
        )

        control_video = torch.cat((v1_context, filler, v2_context), dim=0)
        mask = torch.zeros(
            (control_video.shape[0], height, width),
            dtype=video_1.dtype,
            device=video_1.device,
        )
        mask[context_frames:context_frames + filler_count] = 1.0

        start_images = _trim_outer(video_1, trim, from_start=False)
        end_images = _trim_outer(video_2, trim, from_start=True)
        length = int(control_video.shape[0])

        if debug:
            print("\n[VACEJoinPrep] === Start ===")
            print(f"[VACEJoinPrep] video_1 frames: {video_1.shape[0]}")
            print(f"[VACEJoinPrep] video_2 frames: {video_2.shape[0]}")
            print(f"[VACEJoinPrep] size: {width}x{height}")
            print(f"[VACEJoinPrep] context_frames: {context_frames}")
            print(f"[VACEJoinPrep] replace_frames: {replace_frames}")
            print(f"[VACEJoinPrep] new_frames: {new_frames}")
            print(f"[VACEJoinPrep] control length: {length}")
            print(f"[VACEJoinPrep] start_images: {start_images.shape[0]}")
            print(f"[VACEJoinPrep] end_images: {end_images.shape[0]}")
            print("[VACEJoinPrep] === End ===")

        return (
            control_video,
            mask,
            width,
            height,
            length,
            start_images,
            end_images,
            context_frames,
            replace_frames,
            new_frames,
        )


class VACECrossfadeTransition:
    EASING = ["linear", "ease_in", "ease_out", "ease_in_out"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_video": ("IMAGE",),
                "vace_output": ("IMAGE",),
                "context_frames": ("INT", {"default": 8, "min": 1, "max": 4096}),
                "easing": (cls.EASING, {"default": "ease_in"}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "build"
    CATEGORY = "video/VACE"
    DESCRIPTION = "Blend the original seam context into the regenerated VACE seam and return the transition clip."

    def build(self, control_video, vace_output, context_frames, easing, enabled):
        _validate_video_tensor("control_video", control_video)
        _validate_video_tensor("vace_output", vace_output)

        if control_video.shape != vace_output.shape:
            raise ValueError(
                f"control_video and vace_output must match exactly, got "
                f"{tuple(control_video.shape)} vs {tuple(vace_output.shape)}"
            )

        if not enabled or context_frames <= 0:
            return (vace_output,)

        if control_video.shape[0] < context_frames * 2:
            raise ValueError(
                f"Need at least {context_frames * 2} frames for context blending, got {control_video.shape[0]}"
            )

        alpha = torch.linspace(0.0, 1.0, context_frames, dtype=vace_output.dtype, device=vace_output.device)
        alpha = _apply_easing(alpha, easing).view(-1, 1, 1, 1)

        head_original = control_video[:context_frames]
        head_vace = vace_output[:context_frames]
        head = head_original * (1.0 - alpha) + head_vace * alpha

        tail_vace = vace_output[-context_frames:]
        tail_original = control_video[-context_frames:]
        tail = tail_vace * (1.0 - alpha) + tail_original * alpha

        middle = vace_output[context_frames:-context_frames]
        parts = [head]
        if middle.shape[0] > 0:
            parts.append(middle)
        parts.append(tail)

        return (torch.cat(parts, dim=0),)


NODE_CLASS_MAPPINGS = {
    "VACEClipList3": VACEClipList3,
    "VACEClipLoopStart": VACEClipLoopStart,
    "VACEClipLoopEnd": VACEClipLoopEnd,
    "VACESeedInt": VACESeedInt,
    "VACEJoinPrep": VACEJoinPrep,
    "VACECrossfadeTransition": VACECrossfadeTransition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VACEClipList3": "VACE Clip List (Up To 3)",
    "VACEClipLoopStart": "VACE Clip Loop Start",
    "VACEClipLoopEnd": "VACE Clip Loop End",
    "VACESeedInt": "VACE Seed Int",
    "VACEJoinPrep": "VACE Join Prep",
    "VACECrossfadeTransition": "VACE Crossfade Transition",
}
