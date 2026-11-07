from typing_extensions import override
from comfy_api.latest import ComfyExtension
from .nodes import images_and_masks, qq_nodes_option_load

# 注册
class ReplenishNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            *images_and_masks.NODES,
            *qq_nodes_option_load.NODES,
            # *video_nodes.NODES
        ]

async def comfy_entrypoint():
    return ReplenishNodesExtension()