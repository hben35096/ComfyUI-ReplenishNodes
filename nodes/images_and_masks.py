from comfy_api.latest import io, ui
import numpy as np
import torch
from PIL import Image
# import cv2
import os
import random
import folder_paths
from server import PromptServer
from ..function import image_process

# ##################################################

class RN_BatchImageBlend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_BatchImageBlend",
            display_name="Batch Image Blend",
            category="Replenish/Image",
            search_aliases=["Image overlay", ],
            inputs=[
                io.Image.Input("images", optional=False,),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, images, opacity):
        print("input image number:", len(images))

        max_height = max(image.shape[0] for image in images)
        max_width = max(image.shape[1] for image in images)
        print(f"Max width: {max_width}; Max height:, {max_height}")

        # 创建一个透明背景的输出图像
        output_image = Image.new("RGBA", (max_width, max_height), (255, 0, 0, 0))

        for image in images:
            # print("input tensor space:", image.shape) # 调试
            img = image_process.tensor2pil(image)
            img = img.convert("RGBA") 

            # 计算中心位置
            
            alpha_channel = img.split()[-1]  # 获取图像的透明通道
            alpha_channel = Image.eval(alpha_channel, lambda a: int(a * opacity))  # 将原有透明度与 blend_percentage 相乘

            # 替换图像的透明通道
            img.putalpha(alpha_channel)

            output_image = Image.alpha_composite(output_image, img)

        output_tensor = image_process.pil2tensor(output_image)
        # print("Output tensor shape:", output_tensor.shape) # 调试

        return io.NodeOutput(output_tensor, )
    

class RN_MultipleImageBlend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image"),  # template for each input
            prefix="image",                  # prefix for generated input names
            min=2,                           # minimum number of inputs shown
            max=50,                          # maximum number of inputs allowed
        )
        return io.Schema(
            node_id="RN_MultipleImageBlend",
            display_name="Multiple Image Blend",
            category="Replenish/Image",
            search_aliases=["Image overlay", ],
            description="Overlay multiple images without changing the size of the source images and output an RGBA four-channel image.",
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type, opacity) -> io.NodeOutput:
        # 'images' is a dict mapping input names to their values
        image_list = list(images.values())
        print("input image number:", len(image_list))

        max_height = max(image.shape[1] for image in image_list)
        max_width = max(image.shape[2] for image in image_list)
        print(f"Max width: {max_width}; Max height: {max_height}")

        # 创建一个透明背景的输出图像的 NumPy 数组
        output_image = np.zeros((max_height, max_width, 4), dtype=np.float32)

        for image in image_list:
            # 转换张量到 PIL 图像并确保是 RGBA 模式
            img = image_process.tensor2pil(image).convert("RGBA")
            img = np.array(img, dtype=np.float32)  # 转换为 NumPy 数组以便操作

            # 计算中心位置
            x_pos = (max_width - img.shape[1]) // 2
            y_pos = (max_height - img.shape[0]) // 2
            
            # 调用 blend_layers 函数叠加图像
            output_image[y_pos:y_pos+img.shape[0], x_pos:x_pos+img.shape[1]] = \
                cls.blend_layers(output_image[y_pos:y_pos+img.shape[0], x_pos:x_pos+img.shape[1]], img, opacity)

        # 将结果转换为 PIL 图像并转换为张量格式
        output_image_pil = Image.fromarray(np.clip(output_image, 0, 255).astype(np.uint8), 'RGBA')
        output_tensor = image_process.pil2tensor(output_image_pil)

        return io.NodeOutput(output_tensor, )

    # @staticmethod
    def blend_layers(base, overlay, opacity):
        alpha_overlay = overlay[..., 3] * opacity / 255.0  # 叠加层的透明度（0到1）
        alpha_base = base[..., 3] / 255.0  # 底层的透明度（0到1）

        # 最终的透明度
        out_alpha = alpha_overlay + alpha_base * (1 - alpha_overlay)

        # 避免除以零的情况
        out_alpha_safe = np.where(out_alpha == 0, 1, out_alpha)

        # 混合 RGB 通道
        for c in range(3):  # R, G, B channels
            base[..., c] = np.where(
                out_alpha == 0,
                base[..., c],  # 如果 `out_alpha` 为零，保持底层像素的值
                (overlay[..., c] * alpha_overlay + base[..., c] * alpha_base * (1 - alpha_overlay)) / out_alpha_safe
            )

        base[..., 3] = out_alpha * 255  # 更新 alpha 通道

        return base

# 根据参考图改变图片大小，输出是 RGBA
class RN_Reference_Resize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image"),  # template for each input
            prefix="image",                  # prefix for generated input names
            min=1,                           # minimum number of inputs shown
            max=50,                          # maximum number of inputs allowed
        )
        return io.Schema(
            node_id="RN_Reference_Resize",
            display_name="Reference Resize",
            category="Replenish/Image",
            search_aliases=["Image overlay", ],
            description="The source image is not cropped; instead, it is extended with a transparent background. The output RGBA four-channel image has exactly the same dimensions as the reference image.",
            inputs=[
                io.Image.Input("reference_image", ),
                io.Autogrow.Input("images", template=autogrow_template),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )
        
    @classmethod
    def execute(cls, reference_image, images):
        image_list = list(images.values())
        output_tensors = []
    
        reference_image = reference_image[0]
        ref_image_pil = image_process.tensor2pil(reference_image).convert("RGBA")
        ref_width, ref_height = ref_image_pil.size
        ref_aspect_ratio = ref_width / ref_height
    
        for img in image_list:
            img_pil = image_process.tensor2pil(img).convert("RGBA")
            img_width, img_height = img_pil.size
            aspect_ratio = img_width / img_height
    
            if aspect_ratio < ref_aspect_ratio:
                new_height = ref_height
                new_width = int(ref_height * aspect_ratio)
            else:
                new_width = ref_width
                new_height = int(ref_width / aspect_ratio)
    
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
    
            # 创建与参考图相同尺寸的透明画布
            canvas = Image.new("RGBA", (ref_width, ref_height), (0, 0, 0, 0))
    
            # 居中粘贴
            x = (ref_width - new_width) // 2
            y = (ref_height - new_height) // 2
            canvas.paste(img_pil, (x, y))
    
            output_tensor = image_process.pil2tensor(canvas)
            output_tensors.append(output_tensor)
    
        batch_output_tensor = torch.cat(output_tensors, dim=0)
    
        return io.NodeOutput(batch_output_tensor)

# 前后景对齐
class RN_BatchImageAlign(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_BatchImageAlign",
            display_name="Batch Image Align",
            category="Replenish/Image",
            search_aliases=["Image overlay", ],
            inputs=[
                io.Image.Input("image_bg", ),
                io.Image.Input("images_fg", ),
                io.Combo.Input("align", options=["◤ top left", "top center ▼", "top right ◥", "◣ bottom left", "bottom center ▲", "bottom right ◢", "center ◆"],),
                io.Int.Input("offset_x", default=0, min=-2048, max=2048, step=1),
                io.Int.Input("offset_y", default=0, min=-2048, max=2048, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="merge_image"),
                io.Image.Output(display_name="only_fg"),
            ],
        )

    @classmethod
    def execute(cls, image_bg, images_fg, align, offset_x, offset_y) -> io.NodeOutput:
        output_tensors = []
        empty_tensors = []
        image_bg_pil = image_process.tensor2pil(image_bg).convert("RGBA")
        bg_width, bg_height = image_bg_pil.size
        x, y = 0, 0 
        for img in images_fg:
            img_pil = image_process.tensor2pil(img).convert("RGBA")
            width, height = img_pil.size
            if align == "◤ top left":
                x = 0 + offset_x
                y = 0 + offset_y
            if align == "top center ▼":
                x = (bg_width - width) // 2 + offset_x
                y = 0 + offset_y
            elif align == "top right ◥":
                x = bg_width - width + offset_x
                y = 0 + offset_y
            elif align == "◣ bottom left":
                x = 0 + offset_x
                y = bg_height - height + offset_y
            elif align == "bottom center ▲":
                x = (bg_width - width) // 2 + offset_x
                y = bg_height - height + offset_y
            elif align == "bottom right ◢":
                x = bg_width - width + offset_x
                y = bg_height - height + offset_y
            elif align == "center ◆":
                x = (bg_width - width) // 2 + offset_x
                y = (bg_height - height) // 2 + offset_y

            new_image_pil = image_bg_pil
            new_image_pil.paste(img_pil, (x, y), img_pil)  # 原地粘贴，别赋值
            output_tensor = image_process.pil2tensor(new_image_pil)
            output_tensors.append(output_tensor)

            empty_img = Image.new("RGBA", (bg_width, bg_height), (0, 0, 0, 0))  # 全透明背景
            empty_img.paste(img_pil, (x, y), img_pil)
            empty_img_tensor = image_process.pil2tensor(empty_img)
            empty_tensors.append(empty_img_tensor)
            
        batch_output_tensor = torch.cat(output_tensors, dim=0) # 返回批量图像张量
        empty_batch_output_tensor = torch.cat(empty_tensors, dim=0) 

        return io.NodeOutput(batch_output_tensor, empty_batch_output_tensor)


class RN_ImageBlendBG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_ImageBlendBG",
            display_name="Image Blend BG",
            category="Replenish/Image",
            search_aliases=["Image overlay", ],
            inputs=[
                io.Image.Input("image_bg", ),
                io.Image.Input("image", ),
                io.Combo.Input("align", options=["size", "top left", "center"],),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01),

            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, image_bg, image, align, opacity) -> io.NodeOutput:
        image_bg_pil = image_process.tensor2pil(image_bg).convert("RGBA")
        img_pil = image_process.tensor2pil(image).convert("RGBA")
        
        output_image_pil = image_process.image_blend(image_bg_pil, img_pil, align, opacity)

        output_tensor = image_process.pil2tensor(output_image_pil)

        # 返回最终图像
        return io.NodeOutput(output_tensor,)
        


# 在不裁剪源图像尺寸的前提下叠加多张图像，可以选择不同的对齐模式，输出一张 RGBA 四通道图像。
class RN_MultipleImageBlend_2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image"),  # template for each input
            prefix="image",                  # prefix for generated input names
            min=2,                           # minimum number of inputs shown
            max=50,                          # maximum number of inputs allowed
        )
        return io.Schema(
            node_id="RN_MultipleImageBlend_2",
            display_name="Multiple Image Blend 2",
            category="Replenish/Image",
            search_aliases=["Image overlay", ],
            description="Multiple images can be overlaid without cropping the source image dimensions, with different alignment modes available, to output a single RGBA four-channel image.",
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template),
                io.Float.Input("opacity", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("align", options=["size", "top left", "center"],),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )
        
    @classmethod
    def execute(cls, images: io.Autogrow.Type, opacity, align) -> io.NodeOutput:
        # 'images' is a dict mapping input names to their values
        image_list = list(images.values())

        print("Input image number:", len(image_list))
        max_height = max(image.shape[1] for image in image_list)
        max_width = max(image.shape[2] for image in image_list)
        print(f"Max width: {max_width}; Max height: {max_height}")
        
        image_bg_pil = Image.new("RGBA", (max_width, max_height), (255, 0, 0, 0)) # 创建一个透明背景的输出图像

        # blender = RN_ImageBlendBG() # 实例化 RN_ImageBlendBG

        for image in image_list:
            img_pil = image_process.tensor2pil(image).convert("RGBA")

            image_bg_pil = image_process.image_blend(image_bg_pil, img_pil, align, opacity) # 使用实例调用 image_blend 方法进行批处理

        output_tensor = image_process.pil2tensor(image_bg_pil)
        return io.NodeOutput(output_tensor,)
    
# 这个基本功能完成了，需要让它更灵活，还要做成批处理 完成
class RN_FillAlpha(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_FillAlpha",
            display_name="Fill Alpha",
            category="Replenish/Image",
            search_aliases=["Image overlay", ],
            inputs=[
                io.Image.Input("images",),
                # io.String.Input("fill_color", default="0,0,0",),
                io.Color.Input("fill_color"),
            ],
            outputs=[
                io.Image.Output(display_name="rgb_image"),
            ],
        )
        
    @classmethod
    def execute(cls, images, fill_color) -> io.NodeOutput:
        # 解析填充颜色
        if fill_color.startswith('#'):
            hex_color = fill_color.lstrip('#')
            fill_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        else:
            fill_color = tuple(map(int, fill_color.split(',')))

        output_tensors = []
        for image in images:
            # print("图片空间:", image.shape) # 调试

            # 创建RGBA背景图像
            height, width = image.shape[0], image.shape[1]
            bg_image = Image.new("RGBA", (width, height), (*fill_color, 255))

            image = image_process.tensor2pil(image).convert("RGBA")

            output_image = Image.alpha_composite(bg_image, image).convert("RGB") # 合成并转换为RGB模式

            # 转换为张量并存储
            output_tensor = image_process.pil2tensor(output_image)
            output_tensors.append(output_tensor)

        # 返回批量图像张量
        batch_output_tensor = torch.cat(output_tensors, dim=0)
        
        return io.NodeOutput(batch_output_tensor,)

class RN_ToRGB(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_ToRGB",
            display_name="To RGB",
            category="Replenish/Image",
            search_aliases=["Channel", ],
            inputs=[
                io.Image.Input("images",),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, images) -> io.NodeOutput:
        output_tensors = []
        for image in images:
            image = image_process.tensor2pil(image).convert("RGB")
            output_tensor = image_process.pil2tensor(image)
            output_tensors.append(output_tensor)
        batch_output_tensor = torch.cat(output_tensors, dim=0) # 返回批量图像张量

        return io.NodeOutput(batch_output_tensor,)
    
# 遮罩色阶调节
class RN_MaskLevelsAdjust(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_MaskLevelsAdjust",
            display_name="Mask Levels Adjust",
            category="Replenish/Image",
            search_aliases=["Image overlay", ],
            inputs=[
                io.Mask.Input("masks", optional=False),
                io.Int.Input("output_black_point", default=0, min=0, max=255, step=1, display_mode=io.NumberDisplay.slider,), # display_mode=io.NumberDisplay.slider,
                io.Int.Input("output_white_point", default=255, min=0, max=255, step=1, display_mode=io.NumberDisplay.slider),
            ],
            outputs=[
                io.Mask.Output(display_name="masks"),
            ],
            is_output_node=True
        )
        
    @classmethod
    def execute(cls, masks, output_black_point, output_white_point) -> io.NodeOutput:
        l_masks = []
        ret_masks = []

        for m in masks:
            l_masks.append(torch.unsqueeze(m, 0))

        for i in range(len(l_masks)):
            _mask = l_masks[i]
            mask_np = np.clip(255. * _mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            orig_mask = Image.fromarray(mask_np, mode="L")

            # 直接对遮罩应用色阶调整
            ret_mask = image_process.adjust_levels(orig_mask, 0, 255, 1, output_black_point, output_white_point)
            region_tensor = image_process.pil2mask(ret_mask).unsqueeze(0)  # 添加批次维度
            ret_masks.append(region_tensor)

        result_tensor = torch.cat(ret_masks, dim=0) 
        print("Number of masks processed:", len(ret_masks))
        # print("自己遮罩张量的形状:", result_tensor.shape)  # 调试

        # return io.NodeOutput(result_tensor,)
        return io.NodeOutput(result_tensor, ui=ui.PreviewMask(result_tensor, cls=cls))

class RN_PreviewImageLow(io.ComfyNode):
    output_dir = folder_paths.get_temp_directory()
    type = "temp"
    prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
    compress_level = 1

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_PreviewImageLow",
            display_name="Preview Image-JPEG",
            category="Replenish/Image",
            search_aliases=["Select image", ],
            description="It can be used for quickly previewing images, as well as for selecting and exporting images.",
            inputs=[
                io.Image.Input("images",),
                io.Float.Input("quality", default=0.5, min=0.0, max=1.0, step=0.01, display_name=f"preview_quality"),
                io.Float.Input("preview_size", default=1, min=0.01, max=1.0, step=0.01),
                io.Int.Input("output_number", default=1, min=1, max=9999, step=1, tooltip="Output a single lossless source image."),
            ],
            outputs=[io.Image.Output(display_name="single_image", tooltip="Source image (Single)."),],
            hidden=[io.Hidden.unique_id],
            is_output_node=True
        )
        
    @classmethod
    def execute(cls, images, quality, preview_size, output_number, filename_prefix="ComfyUI") -> io.NodeOutput:
        filename_prefix += cls.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
        )
        results = list()
        file_total = len(images)
        output_info = ""
        if output_number > file_total:
            output_info = f"output_number {output_number} > Input Images {file_total}, the last one will be selected"
            output_number = file_total
        file_count = 0
        total_size = 0
        output_image = images[0].unsqueeze(0)
        for (batch_number, image) in enumerate(images):
            if output_number == batch_number + 1:
                output_image = image.unsqueeze(0)
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if img.mode == 'RGBA':
                img = img.convert('RGB')  # 去掉透明通道以兼容 JPEG

            if preview_size != 1.0:
                new_size = (max(1, round(img.width * preview_size)), max(1, round(img.height * preview_size)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"  # 保存为 JPG 格式
            quality_int = int(quality * 100)
            file_path = os.path.join(full_output_folder, file)
            img.save(file_path, format="JPEG", quality=quality_int)
    
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": cls.type
            })
            counter += 1
            file_count += 1
            total_size += os.path.getsize(file_path)
            
        if cls.hidden.unique_id:
            file_info = f"input: {file_count} | total size: {image_process.readable_size(total_size)}"
            if output_info:
                file_info = file_info + f"\n{output_info}"
            PromptServer.instance.send_progress_text(
                file_info,
                cls.hidden.unique_id
            )
            
        # return {"ui": {"images": results}}
        return {"ui": {"images": results}, "result": io.NodeOutput(output_image,)}
        # return io.NodeOutput(images, ui=ui.PreviewImage(results, cls=cls))

# 通过索引获取单张图像，如果数值大于输入图像总数，使用最后的图像
class RN_IndexToSingleImage(io.ComfyNode):
    output_dir = folder_paths.get_temp_directory()
    type = "temp"
    prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
    compress_level = 1

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_IndexToSingleImage",
            display_name="Index To Single Image",
            category="Replenish/Image",
            # search_aliases=["", ],
            description="Obtain the specified image by index and output a single image with the original lossless quality, Index 0 is the first image.",
            inputs=[
                io.Image.Input("images",),
                io.Int.Input("index", default=0, min=0, max=9999, step=1, tooltip="Index 0 is the first image."),
                io.Float.Input("preview_quality", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="This value is only used for previewing the image and will not be output."),
            ],
            outputs=[io.Image.Output(display_name="image", tooltip="Source image (Single)."),],
            hidden=[io.Hidden.unique_id],
            is_output_node=True
        )
        
    @classmethod
    def execute(cls, images, index, preview_quality, filename_prefix="ComfyUI") -> io.NodeOutput:
        filename_prefix += cls.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, cls.output_dir, images[0].shape[1], images[0].shape[0]
        )
        results = list()
        file_total = len(images)
        file_info = f"Input Images {file_total}, Index {index}"
        if index > file_total - 1:
            file_info = f"Index {index} > Input Images {file_total}, the last one will be selected"
            index = file_total - 1

        output_image = None
        for (batch_number, image) in enumerate(images):
            if batch_number == index:
                output_image = image.unsqueeze(0)
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                if img.mode == 'RGBA':
                    img = img.convert('RGB')  # 去掉透明通道以兼容 JPEG
                
                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch_num}_{counter:05}_.jpg"  # 保存为 JPG 格式
                quality_int = int(preview_quality * 100)
                file_path = os.path.join(full_output_folder, file)
                img.save(file_path, format="JPEG", quality=quality_int)
        
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": cls.type
                })
                counter += 1
                break
            
        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(
                file_info,
                cls.hidden.unique_id
            )

        return {"ui": {"images": results}, "result": io.NodeOutput(output_image,)}

class RN_ToJPEG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RN_ToJPEG",
            display_name="To JPEG",
            category="Replenish/Image",
            search_aliases=["Image overlay", ],
            inputs=[
                io.Image.Input("images",),
                io.Float.Input("quality", default=0.5, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output(display_name="images"),],
        )
        
    @classmethod
    def execute(cls, images, quality) -> io.NodeOutput:
        from io import BytesIO
        buffer = BytesIO()
        output_tensors = []
        for image in images:
            image = image_process.tensor2pil(image).convert("RGB")
            image.save(buffer, format="JPEG", quality=int(quality * 100))
            buffer.seek(0)
            compressed_image = Image.open(buffer)
            output_tensor = image_process.pil2tensor(compressed_image)
            output_tensors.append(output_tensor)
        batch_output_tensor = torch.cat(output_tensors, dim=0) # 返回批量图像张量

        return io.NodeOutput(batch_output_tensor,)


NODES = [
    RN_BatchImageBlend,
    RN_MultipleImageBlend,
    RN_ImageBlendBG,
    RN_FillAlpha,
    RN_MaskLevelsAdjust,
    RN_MultipleImageBlend_2,
    RN_PreviewImageLow,
    RN_ToRGB,
    RN_ToJPEG,
    RN_Reference_Resize,
    RN_BatchImageAlign,
    RN_IndexToSingleImage,
]
