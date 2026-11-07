from comfy_api.latest import io
import folder_paths
import comfy.samplers

class GetBatchCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetBatchCount",
            display_name="Get Batch Count",
            category="Replenish/QQNodes",
            search_aliases=["QQNodes",],
            description="It is used to obtain the number of pictures in the grid control and convert it into a queue batch value.",
            inputs=[
                io.Custom("XY_GRID_CONTROL").Input("xy_grid_control"),
            ],
            outputs=[
                io.String.Output(display_name="Batch Count Str"),
                io.Int.Output(display_name="Batch Count"),
            ],
        )

    @classmethod
    def execute(cls, xy_grid_control) -> io.NodeOutput:
        batch_count_str = str(xy_grid_control)
        cleaned_str = batch_count_str.strip("()")
        elements = cleaned_str.split(", ")
        first_element = int(elements[0].strip("'"))
        return io.NodeOutput(batch_count_str, first_element,)
    
# 加载LoRA名称
class XYLoadLoraName(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="XYLoadLoraName",
            display_name="Load Lora Name",
            category="Replenish/QQNodes",
            search_aliases=["lora",],
            description="Select a LoRA name and output as a string.",
            inputs=[
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras"), tooltip="The name of the LoRA."),
            ],
            outputs=[
                io.String.Output(display_name="lora_name"),
            ],
        )

    @classmethod
    def execute(cls, lora_name) -> io.NodeOutput:
        return io.NodeOutput(lora_name, )
    
# 加载采样器名称
class XYLoadSamplerName(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="XYLoadSamplerName",
            display_name="Load Sampler Name",
            category="Replenish/QQNodes",
            search_aliases=["lora",],
            description="Select a sampler name and output as a string.",
            inputs=[
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, tooltip="The name of the sampler."),
            ],
            outputs=[
                io.String.Output(display_name="sampler_name"),
            ],
        )


    @classmethod
    def execute(cls, sampler_name) -> io.NodeOutput:
        return io.NodeOutput(sampler_name, )

# 加载调度器名称
class XYLoadSchedulerName(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="XYLoadSchedulerName",
            display_name="Load Scheduler Name",
            category="Replenish/QQNodes",
            search_aliases=["scheduler",],
            description="Get the scheduler name and output as a string",
            inputs=[
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, tooltip="The name of the sampler's scheduler."),
            ],
            outputs=[
                io.String.Output(display_name="scheduler_name"),
            ],
        )
    @classmethod
    def execute(cls, scheduler) -> io.NodeOutput:
        return io.NodeOutput(scheduler, )

# 加载checkpoint名称
class XYLoadCkptName(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="XYLoadCkptName",
            display_name="Load Checkpoint Name",
            category="Replenish/QQNodes",
            search_aliases=["scheduler",],
            description="Select a checkpoint name and output as a string.",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints"), tooltip="The name of the checkpoint (model) to load."),
            ],
            outputs=[
                io.String.Output(display_name="checkpoint_name"),
            ],
        )
    @classmethod
    def execute(cls, ckpt_name) -> io.NodeOutput:
        return io.NodeOutput(ckpt_name, )
    
# 加载 Unet名称
class XYLoadUNETName(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="XYLoadUNETName",
            display_name="Load diffusion_model Name",
            category="Replenish/QQNodes",
            search_aliases=["scheduler",],
            description="Select a diffusion_models(unet) model name and output as a string.",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models"), tooltip="The name of the diffusion_models to load."),
            ],
            outputs=[
                io.String.Output(display_name="diffusion_model_name"),
            ],
        )
        
    @classmethod
    def execute(cls, unet_name) -> io.NodeOutput:
        return io.NodeOutput(unet_name, )
    
# 加载 Unet名称
class XYLoadCLIPName(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="XYLoadCLIPName",
            display_name="Load CLIP Model Name",
            category="Replenish/QQNodes",
            search_aliases=["scheduler",],
            description="Select a CLIP Model name and output as a string.",
            inputs=[
                io.Combo.Input("clip_name", options=folder_paths.get_filename_list("clip"), tooltip="The name of the clip model to load."),
            ],
            outputs=[
                io.String.Output(display_name="clip_model_name"),
            ],
        )

    @classmethod
    def execute(cls, clip_name) -> io.NodeOutput:
        return io.NodeOutput(clip_name, )
    

NODES = [
    GetBatchCount,
    XYLoadLoraName,
    XYLoadSamplerName,
    XYLoadSchedulerName,
    XYLoadCkptName,
    XYLoadUNETName,
    XYLoadCLIPName,
]

