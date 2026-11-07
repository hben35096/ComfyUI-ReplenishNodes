import torch
import math

# 一些暂时不知道要还是不要的函数

_BLACK_FLOOR = 1e-4

# from https://github.com/nikaskeba/ComfyUI-Minimax-H3-Reference-Library
def _mean_luminance(images):
    rgb = images[..., :3].float()
    luminance = (
        rgb[..., 0] * 0.2126 +
        rgb[..., 1] * 0.7152 +
        rgb[..., 2] * 0.0722
    )
    return float(luminance.mean().item())

# from https://github.com/nikaskeba/ComfyUI-Minimax-H3-Reference-Library
def match_boundary_luminance(
    context_frames,
    target_frames,
    analysis_frames=4,
    correction_frames=8,
    strength=1.0,
    max_adjustment_ev=0.25,
):
    """Match target exposure against the ending context frames."""

    if (
        float(strength) <= 0.0 or
        context_frames.ndim != 4 or
        target_frames.ndim != 4 or
        context_frames.shape[-1] < 3 or
        target_frames.shape[-1] < 3 or
        context_frames.shape[0] == 0 or
        target_frames.shape[0] == 0
    ):
        return target_frames

    analysis = min(max(1, int(analysis_frames)), context_frames.shape[0])
    count = min(max(1, int(correction_frames)), target_frames.shape[0])

    reference_luminance = _mean_luminance(
        context_frames[-analysis:]
    )

    if reference_luminance <= _BLACK_FLOOR:
        return target_frames

    rgb = target_frames[:count, ..., :3].float()

    luminance = (
        rgb[..., 0] * 0.2126 +
        rgb[..., 1] * 0.7152 +
        rgb[..., 2] * 0.0722
    )

    frame_luminance = luminance.mean(dim=(1, 2))
    safe_luminance = frame_luminance.clamp_min(_BLACK_FLOOR)

    reference = torch.as_tensor(
        reference_luminance,
        device=target_frames.device,
        dtype=torch.float32,
    )

    exposure_ev = torch.log2(reference / safe_luminance)

    limit = max(0.0, float(max_adjustment_ev))
    exposure_ev = exposure_ev.clamp(-limit, limit)

    exposure_ev = torch.where(
        frame_luminance > _BLACK_FLOOR,
        exposure_ev,
        torch.zeros_like(exposure_ev),
    )

    if count == 1:
        weights = torch.ones(
            1, device=target_frames.device, dtype=torch.float32
        )
    else:
        positions = torch.arange(
            count, device=target_frames.device, dtype=torch.float32
        )
        weights = 0.5 * (
            1.0 + torch.cos(
                math.pi * positions / float(count - 1)
            )
        )

    gains = torch.exp2(exposure_ev * float(strength) * weights)
    gains = gains.to(dtype=target_frames.dtype)

    output = target_frames.clone()
    output[:count] = (
        output[:count] * gains[:, None, None, None]
    ).clamp(0.0, 1.0)

    return output