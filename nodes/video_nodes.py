"""Conservative luminance matching for decoded clip joins."""
from comfy_api.latest import io
from comfy_api.latest._input_impl.video_types import VideoFromFile

from ..function import spare

# 效果并不是很理想，MiniMax H3 后段视频的问题，不只是亮度变化
class RN_ExposureMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_ExposureMatch",
            display_name="Exposure Match",
            category="Replenish/Image",
            search_aliases=["video", ],
            inputs=[
                io.MultiType.Input("previous_video", types=[io.Image, io.Video]),
                io.MultiType.Input("target_video", types=[io.Image, io.Video]),
                io.Int.Input("analysis_frames", default=6, min=1, max=240, step=1,),
                io.Float.Input("strength", default=0.75, min=0.0, max=10.0, step=0.05, tooltip="0 disables matching. Lower values preserve "),
                io.Float.Input("max_adjustment_ev", default=0.25, min=0.0, max=10.0, step=0.05, tooltip="0 disables matching. Lower values preserve "),
                io.Int.Input("fade_frames", default=24, min=1, max=240, step=1,),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )



    @classmethod
    def execute(cls, previous_video, target_video, analysis_frames, strength, max_adjustment_ev, fade_frames) -> io.NodeOutput:
        if isinstance(previous_video, VideoFromFile):
            previous_video = previous_video.get_components().images
        if isinstance(target_video, VideoFromFile):
            target_video = target_video.get_components().images

        target_frames = spare.match_boundary_luminance(
            context_frames=previous_video,
            target_frames=target_video,
            analysis_frames=analysis_frames,
            correction_frames=fade_frames,
            strength=strength,
            max_adjustment_ev=max_adjustment_ev,
        )
        
        return io.NodeOutput(target_frames,)


NODES = [
    RN_ExposureMatch
]