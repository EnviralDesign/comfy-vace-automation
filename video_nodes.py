from __future__ import annotations

import os
from pathlib import Path

import folder_paths
from comfy_api.latest import InputImpl, io


def _candidate_paths(path_value, base_directory=None):
    path_text = os.path.expandvars(str(path_value or "").strip().strip("\"'"))
    if not path_text:
        raise ValueError("path must not be empty")

    path = Path(path_text).expanduser()
    if path.is_absolute():
        return [path]

    candidates = []
    base_text = os.path.expandvars(str(base_directory or "").strip().strip("\"'"))
    if base_text:
        candidates.append(Path(base_text).expanduser() / path)

    candidates.append(Path(folder_paths.get_input_directory()) / path)
    candidates.append(Path.cwd() / path)

    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = str(resolved).casefold()
        if key not in seen:
            unique.append(resolved)
            seen.add(key)
    return unique


def _resolve_video_path(path_value, base_directory=None):
    candidates = _candidate_paths(path_value, base_directory)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    formatted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Video path does not exist. Checked: {formatted}")


class VACELoadVideoFromPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VACELoadVideoFromPath",
            display_name="VACE Load Video From Path",
            category="video/VACE",
            description=(
                "Load a native Comfy VIDEO from an absolute path, or from a relative "
                "path resolved against base_directory, Comfy's input directory, then "
                "the Comfy process working directory."
            ),
            inputs=[
                io.String.Input(
                    "path",
                    default="",
                    multiline=False,
                    tooltip="Absolute video path or relative video path.",
                ),
                io.String.Input(
                    "base_directory",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Optional base directory used when path is relative.",
                ),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
            ],
        )

    @classmethod
    def validate_inputs(cls, path, base_directory=""):
        if path is None:
            return True
        try:
            _resolve_video_path(path, base_directory)
        except Exception as err:
            return str(err)
        return True

    @classmethod
    def fingerprint_inputs(cls, path, base_directory=""):
        if path is None:
            return None
        video_path = _resolve_video_path(path, base_directory)
        stat = video_path.stat()
        return (str(video_path), stat.st_mtime_ns, stat.st_size)

    @classmethod
    def execute(cls, path, base_directory="") -> io.NodeOutput:
        video_path = _resolve_video_path(path, base_directory)
        return io.NodeOutput(InputImpl.VideoFromFile(str(video_path)))


__all__ = ["VACELoadVideoFromPath"]
