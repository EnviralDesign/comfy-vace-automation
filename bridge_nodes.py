from __future__ import annotations

from fractions import Fraction

import torch
from comfy_api.latest import InputImpl, Types, io


MAX_VISIBLE_BRIDGE_FRAMES = 80
MAX_WAN_GENERATION_FRAMES = 81
EDGE_BLEND_EASING = ["linear", "ease_in", "ease_out", "ease_in_out"]


def _validate_image_batch(name, images):
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if images.ndim != 4:
        raise ValueError(f"{name} must be IMAGE batch data shaped [frames, height, width, channels]")
    if images.shape[-1] < 3:
        raise ValueError(f"{name} must have at least 3 channels")


def _empty_like_video(images):
    return images[:0]


def _cat_nonempty(parts, fallback):
    nonempty = [part for part in parts if part.shape[0] > 0]
    if not nonempty:
        return _empty_like_video(fallback)
    if len(nonempty) == 1:
        return nonempty[0]
    return torch.cat(nonempty, dim=0)


def _wan_length_for_visible(visible_frames):
    if visible_frames < 1:
        raise ValueError("Bridge span must be at least 1 frame")
    if visible_frames > MAX_VISIBLE_BRIDGE_FRAMES:
        raise ValueError(
            f"Bridge span is {visible_frames} frames, but the visible maximum is "
            f"{MAX_VISIBLE_BRIDGE_FRAMES} frames for an {MAX_WAN_GENERATION_FRAMES}-frame Wan window"
        )
    return ((visible_frames + 3) // 4) * 4 + 1


def _coerce_fraction(value):
    if isinstance(value, Fraction):
        return value
    return Fraction(round(float(value) * 1000), 1000)


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
    raise ValueError(f"Unsupported edge blend easing mode: {easing}")


def _blend_edges(generated, bridge_source, left_edge_frames, right_edge_frames, easing):
    bridge = generated.clone()
    total = bridge.shape[0]

    left_count = min(int(left_edge_frames), total, bridge_source.shape[0])
    if left_count > 0:
        alpha = torch.linspace(0.0, 1.0, left_count, dtype=bridge.dtype, device=bridge.device).view(-1, 1, 1, 1)
        alpha = _apply_easing(alpha, easing)
        bridge[:left_count] = bridge_source[:left_count] * (1.0 - alpha) + bridge[:left_count] * alpha

    right_count = min(int(right_edge_frames), total, bridge_source.shape[0])
    if right_count > 0:
        alpha = torch.linspace(0.0, 1.0, right_count, dtype=bridge.dtype, device=bridge.device).view(-1, 1, 1, 1)
        alpha = _apply_easing(alpha, easing)
        source_tail = bridge_source[-right_count:].to(device=bridge.device, dtype=bridge.dtype)
        bridge_tail = bridge[-right_count:]
        bridge[-right_count:] = bridge_tail * (1.0 - alpha) + source_tail * alpha

    return bridge


class VACETwoVideoBridgePrep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VACETwoVideoBridgePrep",
            display_name="VACE Two Video Bridge Prep",
            category="video/VACE",
            description=(
                "Prepare a direct left/right VACE bridge window for NLE-style seam generation. "
                "The public bridge span is left_replace_frames + right_replace_frames; hidden "
                "Wan padding is inserted at the seam and trimmed by the assemble node."
            ),
            inputs=[
                io.Video.Input("left_video"),
                io.Video.Input("right_video"),
                io.Int.Input("left_replace_frames", default=16, min=0, max=MAX_VISIBLE_BRIDGE_FRAMES),
                io.Int.Input("right_replace_frames", default=16, min=0, max=MAX_VISIBLE_BRIDGE_FRAMES),
                io.Int.Input(
                    "edge_blend_frames",
                    default=4,
                    min=0,
                    max=MAX_VISIBLE_BRIDGE_FRAMES,
                    tooltip="Frames at each outer bridge edge used as original guidance and final blend anchors.",
                ),
                io.Boolean.Input("debug", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="control_video"),
                io.Mask.Output(display_name="control_mask"),
                io.Image.Output(display_name="left_kept"),
                io.Image.Output(display_name="right_kept"),
                io.Image.Output(display_name="bridge_source"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="wan_length"),
                io.Int.Output(display_name="bridge_frames"),
                io.Int.Output(display_name="padding_start"),
                io.Int.Output(display_name="padding_frames"),
                io.Int.Output(display_name="left_edge_frames"),
                io.Int.Output(display_name="right_edge_frames"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="bit_depth"),
            ],
        )

    @classmethod
    def execute(
        cls,
        left_video,
        right_video,
        left_replace_frames: int,
        right_replace_frames: int,
        edge_blend_frames: int,
        debug: bool = False,
    ) -> io.NodeOutput:
        left_components = left_video.get_components()
        right_components = right_video.get_components()
        left_images = left_components.images
        right_images = right_components.images

        _validate_image_batch("left_video.images", left_images)
        _validate_image_batch("right_video.images", right_images)

        if left_images.shape[1:] != right_images.shape[1:]:
            raise ValueError(
                f"Video resolution/channels mismatch: left={tuple(left_images.shape[1:])}, "
                f"right={tuple(right_images.shape[1:])}"
            )

        fps = float(left_components.frame_rate)
        right_fps = float(right_components.frame_rate)
        if abs(fps - right_fps) > 1e-6:
            raise ValueError(f"Both videos must share FPS, left={fps} vs right={right_fps}")

        left_replace = int(left_replace_frames)
        right_replace = int(right_replace_frames)
        if left_replace < 0 or right_replace < 0:
            raise ValueError("Replace frame counts must be non-negative")
        if left_replace > left_images.shape[0]:
            raise ValueError(f"left_replace_frames {left_replace} exceeds left video length {left_images.shape[0]}")
        if right_replace > right_images.shape[0]:
            raise ValueError(f"right_replace_frames {right_replace} exceeds right video length {right_images.shape[0]}")

        bridge_frames = left_replace + right_replace
        wan_length = _wan_length_for_visible(bridge_frames)
        padding_frames = wan_length - bridge_frames
        padding_start = left_replace

        height = int(left_images.shape[1])
        width = int(left_images.shape[2])
        if width % 16 != 0 or height % 16 != 0:
            raise ValueError(f"Video dimensions must be divisible by 16, got {width}x{height}")

        left_kept = left_images[:-left_replace] if left_replace > 0 else left_images
        right_kept = right_images[right_replace:] if right_replace > 0 else right_images
        left_span = left_images[-left_replace:] if left_replace > 0 else _empty_like_video(left_images)
        right_span = right_images[:right_replace] if right_replace > 0 else _empty_like_video(right_images)
        bridge_source = _cat_nonempty([left_span, right_span], left_images)

        if bridge_source.shape[0] != bridge_frames:
            raise RuntimeError(
                f"Internal bridge source length mismatch: expected {bridge_frames}, "
                f"got {bridge_source.shape[0]}"
            )

        control_video = torch.full(
            (wan_length, height, width, left_images.shape[3]),
            0.5,
            dtype=left_images.dtype,
            device=left_images.device,
        )
        control_mask = torch.ones(
            (wan_length, height, width),
            dtype=left_images.dtype,
            device=left_images.device,
        )

        edge_frames = max(0, int(edge_blend_frames))
        left_edge_frames = min(edge_frames, left_replace)
        right_edge_frames = min(edge_frames, right_replace)

        if left_edge_frames > 0:
            control_video[:left_edge_frames] = left_span[:left_edge_frames]
            control_mask[:left_edge_frames] = 0.0

        if right_edge_frames > 0:
            control_video[-right_edge_frames:] = right_span[-right_edge_frames:]
            control_mask[-right_edge_frames:] = 0.0

        bit_depth = max(int(left_video.get_bit_depth()), int(right_video.get_bit_depth()))

        if debug:
            print("\n[VACETwoVideoBridgePrep] === Start ===")
            print(f"[VACETwoVideoBridgePrep] left frames: {left_images.shape[0]}")
            print(f"[VACETwoVideoBridgePrep] right frames: {right_images.shape[0]}")
            print(f"[VACETwoVideoBridgePrep] size: {width}x{height}")
            print(f"[VACETwoVideoBridgePrep] fps: {fps}")
            print(f"[VACETwoVideoBridgePrep] left_replace_frames: {left_replace}")
            print(f"[VACETwoVideoBridgePrep] right_replace_frames: {right_replace}")
            print(f"[VACETwoVideoBridgePrep] bridge_frames visible: {bridge_frames}")
            print(f"[VACETwoVideoBridgePrep] wan_length generated: {wan_length}")
            print(f"[VACETwoVideoBridgePrep] padding_start: {padding_start}")
            print(f"[VACETwoVideoBridgePrep] padding_frames: {padding_frames}")
            print(f"[VACETwoVideoBridgePrep] left_edge_frames: {left_edge_frames}")
            print(f"[VACETwoVideoBridgePrep] right_edge_frames: {right_edge_frames}")
            print("[VACETwoVideoBridgePrep] === End ===")

        return io.NodeOutput(
            control_video,
            control_mask,
            left_kept,
            right_kept,
            bridge_source,
            width,
            height,
            wan_length,
            bridge_frames,
            padding_start,
            padding_frames,
            left_edge_frames,
            right_edge_frames,
            fps,
            bit_depth,
        )


class VACETwoVideoBridgeAssemble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VACETwoVideoBridgeAssemble",
            display_name="VACE Two Video Bridge Assemble",
            category="video/VACE",
            description=(
                "Trim hidden Wan padding from a generated bridge, blend bridge edges against original "
                "source frames, and output both bridge-only and full joined clips."
            ),
            inputs=[
                io.Image.Input("generated_bridge"),
                io.Image.Input("left_kept"),
                io.Image.Input("right_kept"),
                io.Image.Input("bridge_source"),
                io.Int.Input("padding_start", min=0, max=MAX_WAN_GENERATION_FRAMES, force_input=True),
                io.Int.Input("padding_frames", min=0, max=MAX_WAN_GENERATION_FRAMES, force_input=True),
                io.Int.Input("left_edge_frames", min=0, max=MAX_VISIBLE_BRIDGE_FRAMES, force_input=True),
                io.Int.Input("right_edge_frames", min=0, max=MAX_VISIBLE_BRIDGE_FRAMES, force_input=True),
                io.Float.Input("fps", min=0.01, max=1000.0, force_input=True),
                io.Int.Input("bit_depth", min=8, max=10, default=8, step=2, force_input=True),
                io.Boolean.Input("debug", default=False),
                io.Combo.Input(
                    "edge_blend_easing",
                    options=EDGE_BLEND_EASING,
                    default="ease_in",
                    tooltip="Easing curve for blending generated bridge edges back into original frames.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="bridge_images"),
                io.Image.Output(display_name="joined_images"),
                io.Video.Output(display_name="bridge_video"),
                io.Video.Output(display_name="joined_video"),
                io.Float.Output(display_name="fps"),
            ],
        )

    @classmethod
    def execute(
        cls,
        generated_bridge,
        left_kept,
        right_kept,
        bridge_source,
        padding_start: int,
        padding_frames: int,
        left_edge_frames: int,
        right_edge_frames: int,
        fps: float,
        bit_depth: int = 8,
        debug: bool = False,
        edge_blend_easing: str = "ease_in",
    ) -> io.NodeOutput:
        _validate_image_batch("generated_bridge", generated_bridge)
        _validate_image_batch("left_kept", left_kept)
        _validate_image_batch("right_kept", right_kept)
        _validate_image_batch("bridge_source", bridge_source)

        reference_shape = generated_bridge.shape[1:]
        for name, images in (
            ("left_kept", left_kept),
            ("right_kept", right_kept),
            ("bridge_source", bridge_source),
        ):
            if images.shape[1:] != reference_shape:
                raise ValueError(
                    f"{name} resolution/channels mismatch: {tuple(images.shape[1:])} vs {tuple(reference_shape)}"
                )

        pad_start = int(padding_start)
        pad_count = int(padding_frames)
        if pad_start < 0 or pad_count < 0:
            raise ValueError("padding_start and padding_frames must be non-negative")
        if pad_start + pad_count > generated_bridge.shape[0]:
            raise ValueError(
                f"Cannot drop padding range {pad_start}:{pad_start + pad_count} "
                f"from generated_bridge length {generated_bridge.shape[0]}"
            )

        if pad_count > 0:
            bridge_images = torch.cat(
                (
                    generated_bridge[:pad_start],
                    generated_bridge[pad_start + pad_count:],
                ),
                dim=0,
            )
        else:
            bridge_images = generated_bridge

        if bridge_images.shape[0] != bridge_source.shape[0]:
            raise ValueError(
                f"Visible bridge length mismatch after Wan padding trim: "
                f"generated={bridge_images.shape[0]} vs source={bridge_source.shape[0]}"
            )

        bridge_images = _blend_edges(
            bridge_images,
            bridge_source.to(device=bridge_images.device, dtype=bridge_images.dtype),
            left_edge_frames,
            right_edge_frames,
            edge_blend_easing,
        )
        joined_images = torch.cat((left_kept, bridge_images, right_kept), dim=0)

        frame_rate = _coerce_fraction(fps)
        bridge_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=bridge_images, frame_rate=frame_rate),
            bit_depth=int(bit_depth),
        )
        joined_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=joined_images, frame_rate=frame_rate),
            bit_depth=int(bit_depth),
        )

        if debug:
            print("\n[VACETwoVideoBridgeAssemble] === Start ===")
            print(f"[VACETwoVideoBridgeAssemble] generated frames: {generated_bridge.shape[0]}")
            print(f"[VACETwoVideoBridgeAssemble] padding_start: {pad_start}")
            print(f"[VACETwoVideoBridgeAssemble] padding_frames: {pad_count}")
            print(f"[VACETwoVideoBridgeAssemble] bridge frames: {bridge_images.shape[0]}")
            print(f"[VACETwoVideoBridgeAssemble] joined frames: {joined_images.shape[0]}")
            print(f"[VACETwoVideoBridgeAssemble] edge_blend_easing: {edge_blend_easing}")
            print(f"[VACETwoVideoBridgeAssemble] fps: {float(frame_rate)}")
            print("[VACETwoVideoBridgeAssemble] === End ===")

        return io.NodeOutput(
            bridge_images,
            joined_images,
            bridge_video,
            joined_video,
            float(frame_rate),
        )


__all__ = [
    "VACETwoVideoBridgePrep",
    "VACETwoVideoBridgeAssemble",
]
