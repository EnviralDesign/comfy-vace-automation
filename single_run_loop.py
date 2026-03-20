import os
import time
import uuid

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


class VACEPairLoopStart:
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
                    "tooltip": "Log loop progress to the console",
                }),
            },
            "optional": {
                "pair_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Internal loop state. Leave unconnected.",
                }),
                "run_token": ("STRING", {
                    "default": "",
                    "tooltip": "Internal loop state. Leave unconnected.",
                }),
            },
        }

    RETURN_TYPES = ("FLOW_CONTROL", "INT", "STRING", "STRING", "STRING", "STRING", "STRING", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("flow", "pair_index", "run_token", "work_dir", "workfile_prefix", "video_1_filename", "video_2_filename", "is_first", "is_last")
    FUNCTION = "loop_start"
    CATEGORY = "video/VACE"
    DESCRIPTION = """
    Single-run loop controller for VACE clip joining. Emits the current pair
    context for one iteration while keeping the run work directory stable across
    the whole prompt execution.
    """
    INPUT_IS_LIST = True

    def loop_start(self, **kwargs):
        raw_input_list = kwargs.get("input_list", [])
        if not isinstance(raw_input_list, list):
            raw_input_list = [raw_input_list]

        input_dir = str(_first(kwargs.get("input_dir"), "") or "").strip()
        input_list = [str(item) for item in raw_input_list if item not in (None, "")]
        project_name = str(_first(kwargs.get("project_name"), "") or "").strip()
        debug = bool(_first(kwargs.get("debug"), False))
        pair_index = int(_first(kwargs.get("pair_index"), 0) or 0)
        run_token = str(_first(kwargs.get("run_token"), "") or "").strip()

        list_length = len(input_list)
        if list_length < 2:
            raise ValueError(f"Need at least 2 videos to create transitions, found {list_length}")

        max_index = list_length - 2
        if pair_index < 0 or pair_index > max_index:
            raise ValueError(f"Pair index {pair_index} out of range (valid: 0-{max_index})")

        if not run_token:
            run_token = _make_run_token()

        work_root = f"{project_name}/vace-work" if project_name else "vace-work"
        work_dir = f"{work_root}/{run_token}"
        workfile_prefix = f"{work_dir}/index{pair_index:03d}"

        is_first = pair_index == 0
        is_last = pair_index == max_index

        video_1_name = input_list[pair_index]
        video_2_name = input_list[pair_index + 1]
        video_1_filename = os.path.join(input_dir, video_1_name) if input_dir else video_1_name
        video_2_filename = os.path.join(input_dir, video_2_name) if input_dir else video_2_name

        if debug:
            print("\n[VACE Pair Loop Start] === Start ===")
            print(f"[VACE Pair Loop Start] Pair index: {pair_index} (videos {pair_index + 1}-{pair_index + 2} of {list_length})")
            print(f"[VACE Pair Loop Start] Run token: {run_token}")
            print(f"[VACE Pair Loop Start] {'[FIRST]' if is_first else ''} {'[LAST]' if is_last else ''}")
            print(f"[VACE Pair Loop Start] Input directory: {input_dir}")
            print(f"[VACE Pair Loop Start] Video 1: {video_1_name}")
            print(f"[VACE Pair Loop Start] Video 2: {video_2_name}")
            print(f"[VACE Pair Loop Start] Work dir: {work_dir}")
            print(f"[VACE Pair Loop Start] === End ===")

        return ("stub", pair_index, run_token, work_dir, workfile_prefix, video_1_filename, video_2_filename, is_first, is_last)


class VACEPairLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "pair_index": ("INT", {"forceInput": True}),
                "run_token": ("STRING", {"forceInput": True}),
                "is_last": ("BOOLEAN", {"forceInput": True}),
                "clip1_prefix": ("STRING", {"forceInput": True}),
                "clip2_filename": ("STRING", {"forceInput": True}),
                "clip3_prefix": ("STRING", {"forceInput": True}),
                "fps": ("FLOAT", {"forceInput": True}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("clip2_filename", "fps")
    FUNCTION = "loop_end"
    CATEGORY = "video/VACE"
    DESCRIPTION = """
    Closes the VACE single-run loop. When the current pair is not the last one,
    this node expands the contained pair-processing subgraph for the next pair
    within the same prompt execution.
    """

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
                if class_type != "VACEPairLoopEnd":
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

    def loop_end(
        self,
        flow,
        pair_index,
        run_token,
        is_last,
        clip1_prefix,
        clip2_filename,
        clip3_prefix,
        fps,
        dynprompt=None,
        unique_id=None,
    ):
        del clip1_prefix
        del clip3_prefix

        if not clip2_filename:
            return (ExecutionBlocker(None), ExecutionBlocker(None))

        if is_last:
            return (clip2_filename, fps)

        if GraphBuilder is None:
            raise RuntimeError("GraphBuilder is unavailable; single-run VACE looping cannot execute")

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
        new_open.set_input("run_token", run_token)

        my_clone = graph.lookup_node("Recurse")
        return {
            "result": (my_clone.out(0), my_clone.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "VACEPairLoopStart": VACEPairLoopStart,
    "VACEPairLoopEnd": VACEPairLoopEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VACEPairLoopStart": "🪐 VACE Pair Loop Start",
    "VACEPairLoopEnd": "🪐 VACE Pair Loop End",
}
