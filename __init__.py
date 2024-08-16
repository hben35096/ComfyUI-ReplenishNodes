import os

from .custom_nodes.clipseg import CLIPSegToMask, CombineSegMasks

NODE_CLASS_MAPPINGS = {
    "CLIPSegToMask":CLIPSegToMask,
    "CombineSegMasks":CombineSegMasks
}
