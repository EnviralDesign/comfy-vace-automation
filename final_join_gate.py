from comfy_execution.graph import ExecutionBlocker


class VACEReadyOutputPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True, "tooltip": "A ready signal. In this workflow, clip2_filename indicates the last transition clip has been written."}),
                "find": ("STRING", {"forceInput": True, "tooltip": "Legacy compatibility input. Ignored."}),
                "replace": ("STRING", {"forceInput": True, "tooltip": "Output directory path to release once the ready signal exists."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "ready_output_path"
    CATEGORY = "video/VACE"
    DESCRIPTION = """
    Releases the output directory path only after a ready-signal string exists.
    This replaces the previous StringReplace hack that used clip2_filename as a
    dependency trigger.
    """

    def ready_output_path(self, string, find, replace):
        del find

        ready_value = (string or "").strip()
        output_path = (replace or "").strip()

        if not ready_value:
            return (ExecutionBlocker(None),)

        if not output_path:
            raise ValueError("Output path is empty")

        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "VACEReadyOutputPath": VACEReadyOutputPath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VACEReadyOutputPath": "🪐 VACE Ready Output Path",
}
