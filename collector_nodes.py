from __future__ import annotations

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class VACEClipCollector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        video_template = io.Autogrow.TemplatePrefix(
            io.Video.Input("video"),
            prefix="video",
            min=2,
            max=20,
        )
        return io.Schema(
            node_id="VACEClipCollector",
            display_name="VACE Clip Collector",
            category="video/VACE",
            description="Collect multiple videos, extract frame batches in-memory, and validate shared FPS and resolution for VACE looping.",
            inputs=[
                io.Boolean.Input("debug", default=False),
                io.Autogrow.Input("videos", template=video_template),
            ],
            outputs=[
                io.Image.Output(display_name="clips", is_output_list=True),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="clip_count"),
            ],
        )

    @classmethod
    def execute(cls, debug: bool, videos: io.Autogrow.Type) -> io.NodeOutput:
        video_items = [video for video in videos.values() if video is not None]
        if len(video_items) < 2:
            raise ValueError(f"Need at least 2 videos, found {len(video_items)}")

        clips = []
        fps_values = []
        reference_shape = None

        for index, video in enumerate(video_items, start=1):
            components = video.get_components()
            clip = components.images
            fps_value = float(components.frame_rate)

            if clip is None:
                raise ValueError(f"Video {index} did not produce image frames")
            if clip.ndim != 4:
                raise ValueError(
                    f"Video {index} frames must be IMAGE batch data shaped [frames, height, width, channels], "
                    f"got ndim={clip.ndim}"
                )

            clip_shape = tuple(clip.shape[1:])
            if reference_shape is None:
                reference_shape = clip_shape
            elif clip_shape != reference_shape:
                raise ValueError(
                    f"All videos must share the same resolution/channels, "
                    f"video_1={reference_shape} vs video_{index}={clip_shape}"
                )

            if fps_values and abs(fps_value - fps_values[0]) > 1e-6:
                raise ValueError(
                    f"All videos must share the same FPS, fps_1={fps_values[0]} vs fps_{index}={fps_value}"
                )

            clips.append(clip)
            fps_values.append(fps_value)

        if debug:
            print("\n[VACEClipCollector] === Start ===")
            print(f"[VACEClipCollector] clip_count: {len(clips)}")
            print(f"[VACEClipCollector] fps: {fps_values[0]}")
            for index, clip in enumerate(clips, start=1):
                print(f"[VACEClipCollector] video_{index}: frames={clip.shape[0]} shape={tuple(clip.shape[1:])}")
            print("[VACEClipCollector] === End ===")

        return io.NodeOutput(clips, fps_values[0], len(clips))


class VACEPhase1Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            VACEClipCollector,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return VACEPhase1Extension()


__all__ = [
    "VACEClipCollector",
    "VACEPhase1Extension",
    "comfy_entrypoint",
]
