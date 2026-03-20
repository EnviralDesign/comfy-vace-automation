import hashlib
import os
import threading
import time
import uuid


_STATE_LOCK = threading.Lock()
_RUN_STATES = {}
_MAX_STATES = 32
_STATE_TTL_SECONDS = 6 * 60 * 60


def _make_signature(input_dir, project_name, input_list):
    payload = "\n".join([input_dir.strip(), project_name.strip(), *input_list])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _make_run_id():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"run-{timestamp}-{uuid.uuid4().hex[:6]}"


def _cleanup_states(now):
    stale_keys = [
        key for key, state in _RUN_STATES.items()
        if now - state["updated_at"] > _STATE_TTL_SECONDS
    ]
    for key in stale_keys:
        _RUN_STATES.pop(key, None)

    if len(_RUN_STATES) <= _MAX_STATES:
        return

    sorted_keys = sorted(
        _RUN_STATES.keys(),
        key=lambda key: _RUN_STATES[key]["updated_at"],
    )
    for key in sorted_keys[:-_MAX_STATES]:
        _RUN_STATES.pop(key, None)


class WanVACEBatchContextAuto:
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
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Raw queue index from the workflow. This node remaps it to a per-run 0-based pair index.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log some details to the console",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("work_dir", "workfile_prefix", "video_1_filename", "video_2_filename", "is_first", "is_last")
    FUNCTION = "setup_context"
    CATEGORY = "video/VACE"
    DESCRIPTION = """
    Establishes iteration context for batch video joins, automatically
    isolating each run into its own work directory and remapping stale
    raw index values onto a fresh 0-based sequence.
    """
    INPUT_IS_LIST = True

    def setup_context(self, **kwargs):
        input_dir = kwargs.get("input_dir", [""])[0].strip()
        input_list = [str(item) for item in kwargs.get("input_list", []) if item not in (None, "")]
        project_name = kwargs.get("project_name", [""])[0].strip()
        raw_index = int(kwargs.get("index", [0])[0])
        debug = bool(kwargs.get("debug", [False])[0])

        list_length = len(input_list)
        if list_length < 2:
            raise ValueError(f"Need at least 2 videos to create transitions, found {list_length}")

        if raw_index < 0:
            raise ValueError(f"Index must be >= 0, got {raw_index}")

        max_index = list_length - 2
        signature = _make_signature(input_dir, project_name, input_list)
        now = time.time()

        with _STATE_LOCK:
            _cleanup_states(now)

            state = _RUN_STATES.get(signature)
            restart_reason = None

            if state is None:
                state = self._new_state(raw_index, max_index, now)
                _RUN_STATES[signature] = state
                restart_reason = "new_state"
            else:
                if raw_index == 0 and state["base_raw_index"] != 0:
                    state = self._new_state(raw_index, max_index, now)
                    _RUN_STATES[signature] = state
                    restart_reason = "manual_reset"
                elif raw_index < state["base_raw_index"]:
                    state = self._new_state(raw_index, max_index, now)
                    _RUN_STATES[signature] = state
                    restart_reason = "raw_index_decreased"
                else:
                    effective_index = raw_index - state["base_raw_index"]
                    if effective_index > max_index:
                        state = self._new_state(raw_index, max_index, now)
                        _RUN_STATES[signature] = state
                        restart_reason = "auto_restart_after_completion"
                    else:
                        state["max_index"] = max_index
                        state["updated_at"] = now

            effective_index = raw_index - state["base_raw_index"]
            if effective_index < 0 or effective_index > max_index:
                raise ValueError(
                    f"Effective index {effective_index} out of range (valid: 0-{max_index}). "
                    f"Raw index={raw_index}, base_raw_index={state['base_raw_index']}"
                )

            state["last_raw_index"] = raw_index
            state["last_effective_index"] = effective_index
            state["updated_at"] = now

        work_root = f"{project_name}/vace-work" if project_name else "vace-work"
        work_dir = f"{work_root}/{state['run_id']}"
        workfile_prefix = f"{work_dir}/index{effective_index:03d}"

        is_first = effective_index == 0
        is_last = effective_index == max_index

        video_1_name = input_list[effective_index]
        video_2_name = input_list[effective_index + 1]
        video_1_filename = os.path.join(input_dir, video_1_name) if input_dir else video_1_name
        video_2_filename = os.path.join(input_dir, video_2_name) if input_dir else video_2_name

        if debug:
            print("\n[VACE Batch Context Auto] === Start ===")
            if restart_reason:
                print(f"[VACE Batch Context Auto] Restart reason: {restart_reason}")
            print(f"[VACE Batch Context Auto] Raw index: {raw_index}")
            print(f"[VACE Batch Context Auto] Base raw index: {state['base_raw_index']}")
            print(f"[VACE Batch Context Auto] Effective index: {effective_index} (videos {effective_index + 1}-{effective_index + 2} of {list_length})")
            print(f"[VACE Batch Context Auto] Run ID: {state['run_id']}")
            print(f"[VACE Batch Context Auto] {'[FIRST]' if is_first else ''} {'[LAST]' if is_last else ''}")
            print(f"[VACE Batch Context Auto] Input directory: {input_dir}")
            print(f"[VACE Batch Context Auto] Video 1: {video_1_name}")
            print(f"[VACE Batch Context Auto] Video 2: {video_2_name}")
            print(f"[VACE Batch Context Auto] Work dir: {work_dir}")
            print(f"[VACE Batch Context Auto] === End ===")

        return (work_dir, workfile_prefix, video_1_filename, video_2_filename, is_first, is_last)

    @staticmethod
    def _new_state(base_raw_index, max_index, now):
        return {
            "run_id": _make_run_id(),
            "base_raw_index": base_raw_index,
            "max_index": max_index,
            "last_raw_index": None,
            "last_effective_index": None,
            "updated_at": now,
        }


NODE_CLASS_MAPPINGS = {
    "WanVACEBatchContextAuto": WanVACEBatchContextAuto,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVACEBatchContextAuto": "🪐 VACE Batch Context (Auto Run)",
}
