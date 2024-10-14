import folder_paths
import comfy.samplers

class GetBatchCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "xy_grid_control": ("XY_GRID_CONTROL", {"forceInput": True})
            }
                }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("STRING", "BatchCount",)
    OUTPUT_TOOLTIPS = ("Details.", "Queue batch values.")
    FUNCTION = "get_batch_count"

    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "It is used to obtain the number of pictures in the grid control and convert it into a queue batch value."

    def get_batch_count(self, xy_grid_control):
        batch_count_str = str(xy_grid_control)
        cleaned_str = batch_count_str.strip("()")
        elements = cleaned_str.split(", ")
        first_element = int(elements[0].strip("'"))
        return (batch_count_str, first_element,)
    
# 加载LoRA名称
class XYLoadLoraName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."})
            }
        }
    
    RETURN_TYPES = ("STRING", )
    OUTPUT_TOOLTIPS = ("The LoRA name.", )
    FUNCTION = "load_lora_name"
    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Select a LoRA name and output as a string."

    def load_lora_name(self, lora_name):
        return (lora_name, )
    
# 加载采样器名称
class XYLoadSamplerName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The name of the sampler."}),
            }
        }

    RETURN_TYPES = ("STRING", )
    OUTPUT_TOOLTIPS = ("The sampler name.",)
    FUNCTION = "load_samplers_name"

    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Select a sampler name and output as a string."

    def load_samplers_name(self, sampler_name):
        return (sampler_name, )

# 加载调度器名称
class XYLoadSchedulerName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The name of the sampler's scheduler."}),
            }
        }

    RETURN_TYPES = ("STRING", )
    OUTPUT_TOOLTIPS = ("The scheduler name.",)
    FUNCTION = "load_scheduler_name"

    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Get the scheduler name and output as a string"

    def load_scheduler_name(self, scheduler):
        return (scheduler, )

# 加载checkpoint名称
class XYLoadCkptName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }

    RETURN_TYPES = ("STRING", )
    OUTPUT_TOOLTIPS = ("The checkpoint name.",)
    FUNCTION = "load_ckpt_name"

    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Select a checkpoint name and output as a string."

    def load_ckpt_name(self, ckpt_name):
        return (ckpt_name, )
    
# 加载 Unet名称
class XYLoadUNETName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
            }
        }

    RETURN_TYPES = ("STRING", )
    OUTPUT_TOOLTIPS = ("The unet name.",)
    FUNCTION = "load_unet_name"

    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Select a unet model name and output as a string."

    def load_unet_name(self, unet_name):
        return (unet_name, )
    
# 加载 Unet名称
class XYLoadCLIPName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip"), ),
            }
        }

    RETURN_TYPES = ("STRING", )
    OUTPUT_TOOLTIPS = ("The clip name.",)
    FUNCTION = "load_clip_name"

    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Select a CLIP Model name and output as a string."

    def load_clip_name(self, clip_name):
        return (clip_name, )
    
class XYMultiLineText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True, "default": "1girl, ", "tooltip": "Multi-line text input."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "multi_line_text"
    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Enter any text."

    def multi_line_text(self, input_text):
        return (input_text,)
    
class XYIntegerOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "integer": ("INT", {"default": 0, "min": 0, "max": 999999999999999, "step": 1, "display": "number"}),
            },
        }

    RETURN_TYPES = ("INT", )
    FUNCTION = "integer_output"
    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Enter any integer."

    def integer_output(self, integer):
        return (integer, )

    
class XYFLOATOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_input": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 999.99, "step": 0.01, "display": "number"}),
            },
        }

    RETURN_TYPES = ("FLOAT", )
    FUNCTION = "float_output"
    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Enter any floating-point number."

    def float_output(self, float_input):
        return (float_input, )

# 节点_种子数值
class XYSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "seed": ("INT", {"default": 0, "min": 0,"max": 0xffffffffffffffff})
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "seed"
    CATEGORY = "Replenish/QQNodes"
    DESCRIPTION = "Enter any integer."

    def seed(self, seed):
        return (int(seed), )

NODE_CLASS_MAPPINGS = {
    "Get Batch Count": GetBatchCount,
    "Load Lora Name": XYLoadLoraName,
    "Load Sampler Name": XYLoadSamplerName,
    "Load Scheduler Name": XYLoadSchedulerName,
    "Load Ckpt Name": XYLoadCkptName,
    "Load UNET Name": XYLoadUNETName,
    "Load CLIP Name": XYLoadCLIPName,
    "Multi Line Text": XYMultiLineText,
    "Integer Output": XYIntegerOutput,
    "FLOAT Output": XYFLOATOutput,
    "Seed Output": XYSeed
}


# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Get Batch Count": "获取批次计数",
#     "Load Lora Name": "获取LoRA名称",
#     "Load Sampler Name": "获取采样器名称",
#     "Load Scheduler Name": "获取调度器名称",
#     "Load Ckpt Name": "获取Checkpoint名称",
#     "Load UNET Name": "获取UNET名称",
#     "Load CLIP Name": "获取CLIP名称",
#     "Multi Line Text": "多行文本输入",
#     "Integer Output": "整数",
#     "FLOAT Output": "浮点数",
#     "XYSeed": "种子"
# }