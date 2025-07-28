import numpy as np
import torch
from PIL import Image
import cv2
import os
import random
import folder_paths

# from was-node-suite-comfyui
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# from ComfyUI_LayerStyle
def pil2cv2(pil_img:Image) -> np.array:
    np_img_array = np.asarray(pil_img)
    return cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)

def cv22pil(cv2_img:np.ndarray) -> Image:
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return mask


def adjust_levels(image:Image, input_black:int=0, input_white:int=255, midtones:float=1.0,
                  output_black:int=0, output_white:int=255) -> Image:

    if input_black == input_white or output_black == output_white:
        return Image.new('RGB', size=image.size, color='gray')

    img = pil2cv2(image).astype(np.float64)

    if input_black > input_white:
        input_black, input_white = input_white, input_black
    if output_black > output_white:
        output_black, output_white = output_white, output_black

    # input_levels remap
    if input_black > 0 or input_white < 255:
        img = 255 * ((img - input_black) / (input_white - input_black))
        img[img < 0] = 0
        img[img > 255] = 255

    # # mid_tone
    if midtones != 1.0:
        img = 255 * np.power(img / 255, 1.0 / midtones)

        img[img < 0] = 0
        img[img > 255] = 255

    # output_levels remap
    if output_black > 0 or output_white < 255:
        img = (img / 255) * (output_white - output_black) + output_black
        img[img < 0] = 0
        img[img > 255] = 255

    img = img.astype(np.uint8)
    return cv22pil(img)


# ##################################################

class RN_BatchImageBlend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),  # 不透明度
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "batch_image_blend"

    CATEGORY = "Replenish/Image"

    def batch_image_blend(self, images, opacity):
        print("input image number:", len(images))

        max_height = max(image.shape[0] for image in images)
        max_width = max(image.shape[1] for image in images)
        print(f"Max width: {max_width}; Max height:, {max_height}")

        # 创建一个透明背景的输出图像
        output_image = Image.new("RGBA", (max_width, max_height), (255, 0, 0, 0))

        for image in images:
            # print("input tensor space:", image.shape) # 调试
            img = tensor2pil(image)
            img = img.convert("RGBA") 

            # 计算中心位置
            
            alpha_channel = img.split()[-1]  # 获取图像的透明通道
            alpha_channel = Image.eval(alpha_channel, lambda a: int(a * opacity))  # 将原有透明度与 blend_percentage 相乘

            # 替换图像的透明通道
            img.putalpha(alpha_channel)

            output_image = Image.alpha_composite(output_image, img)

        output_tensor = pil2tensor(output_image)
        # print("Output tensor shape:", output_tensor.shape) # 调试

        return (output_tensor,)
    

class RN_MultipleImageBlend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),  
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),  # 不透明度
            },
            "optional": {
                "image_b": ("IMAGE",), 
                "image_c": ("IMAGE",), 
                "image_d": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_blend"

    CATEGORY = "Replenish/Image"

    def image_blend(self, image_a, opacity, image_b=None, image_c=None, image_d=None):
        # 创建 images 列表并添加非空的图像
        images = [image_a]
        if image_b is not None:
            images.append(image_b)
        if image_c is not None:
            images.append(image_c)
        if image_d is not None:
            images.append(image_d)

        print("input image number:", len(images))

        max_height = max(image.shape[1] for image in images)
        max_width = max(image.shape[2] for image in images)
        print(f"Max width: {max_width}; Max height: {max_height}")

        # 创建一个透明背景的输出图像的 NumPy 数组
        output_image = np.zeros((max_height, max_width, 4), dtype=np.float32)

        for image in images:
            # 转换张量到 PIL 图像并确保是 RGBA 模式
            img = tensor2pil(image).convert("RGBA")
            img = np.array(img, dtype=np.float32)  # 转换为 NumPy 数组以便操作

            # 计算中心位置
            x_pos = (max_width - img.shape[1]) // 2
            y_pos = (max_height - img.shape[0]) // 2
            
            # 调用 blend_layers 函数叠加图像
            output_image[y_pos:y_pos+img.shape[0], x_pos:x_pos+img.shape[1]] = \
                self.blend_layers(output_image[y_pos:y_pos+img.shape[0], x_pos:x_pos+img.shape[1]], img, opacity)

        # 将结果转换为 PIL 图像并转换为张量格式
        output_image_pil = Image.fromarray(np.clip(output_image, 0, 255).astype(np.uint8), 'RGBA')
        output_tensor = pil2tensor(output_image_pil)

        return (output_tensor,)

    # @staticmethod
    def blend_layers(self, base, overlay, opacity):
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
class RN_Reference_Resize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "images": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "image_resize"

    CATEGORY = "Replenish/Image"
    def image_resize(self, reference_image, images):
        output_tensors = []
        ref_image_pil = tensor2pil(reference_image).convert("RGBA")
        ref_width, ref_height = ref_image_pil.size
        ref_aspect_ratio = ref_width / ref_height
        for img in images:
            img_pil = tensor2pil(img).convert("RGBA")
            img_width, img_height = img_pil.size
            aspect_ratio = img_width / img_height
            
            if aspect_ratio < ref_aspect_ratio:
                new_height = ref_height
                new_width = int(ref_height* aspect_ratio)
            else:
                new_width = ref_width
                new_height = int(ref_width / aspect_ratio)

            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
            output_tensor = pil2tensor(img_pil)
            output_tensors.append(output_tensor)
        batch_output_tensor = torch.cat(output_tensors, dim=0) # 返回批量图像张量
        return (batch_output_tensor,)

class RN_BatchImageAlign:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_bg": ("IMAGE",),
                "images_fg": ("IMAGE",),
                "align": (["◤ top left", "top right ◥", "◣ bottom left", "bottom right ◢", "center ▣"],),
                "offset_x": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1,}),
                "offset_y": ("INT", {"default": 0, "min": -2048, "max": 2048, "step": 1,}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "align_images"

    CATEGORY = "Replenish/Image"
    def align_images(self, image_bg, images_fg, align, offset_x, offset_y):
        output_tensors = []
        image_bg_pil = tensor2pil(image_bg).convert("RGBA")
        bg_width, bg_height = image_bg_pil.size
        x, y = 0, 0 
        for img in images_fg:
            img_pil = tensor2pil(img).convert("RGBA")
            width, height = img_pil.size
            if align == "◤ top left":
                x = 0 + offset_x
                y = 0 + offset_y
            elif align == "top right ◥":
                x = bg_width - width + offset_x
                y = 0 + offset_y
            elif align == "◣ bottom left":
                x = 0 + offset_x
                y = bg_height - height + offset_y
            elif align == "bottom right ◢":
                x = bg_width - width + offset_x
                y = bg_height - height + offset_y
            elif align == "center ▣":
                x = (bg_width - width) // 2 + offset_x
                y = (bg_height - height) // 2 + offset_y

            new_image_pil = image_bg_pil
            new_image_pil.paste(img_pil, (x, y), img_pil)  # 原地粘贴，别赋值
            output_tensor = pil2tensor(new_image_pil)
            output_tensors.append(output_tensor)
            
        batch_output_tensor = torch.cat(output_tensors, dim=0) # 返回批量图像张量
        return (batch_output_tensor,)


class RN_ImageBlendBG:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_bg": ("IMAGE",),
                "image": ("IMAGE",),
                "align": (["size", "top left", "center"],),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "get_image"

    CATEGORY = "Replenish/Image"

    def get_image(self, image_bg, image, align, opacity):
        # print("背景图片空间:", image_bg.shape) # 调试
        
        image_bg_pil = tensor2pil(image_bg).convert("RGBA")
        img_pil = tensor2pil(image).convert("RGBA")
        
        output_image_pil = self.image_blend(image_bg_pil, img_pil, align, opacity)

        output_tensor = pil2tensor(output_image_pil)

        # 返回最终图像
        return (output_tensor,)
        
    def image_blend(self, image_bg_pil, img_pil, align, opacity):
        # 获取图像的尺寸
        bg_width, bg_height = image_bg_pil.size
        img_width, img_height = img_pil.size

        # 调整前景图像尺寸以匹配背景图像尺寸
        if align == "size":
            aspect_ratio = img_width / img_height
            bg_aspect_ratio = bg_width / bg_height

            if aspect_ratio < bg_aspect_ratio: # 似乎 > 改成小于更合理
                new_height = bg_height
                new_width = int(bg_height * aspect_ratio)
            else:
                new_width = bg_width
                new_height = int(bg_width / aspect_ratio)

            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)

            new_img = Image.new("RGBA", (bg_width, bg_height), (0, 0, 0, 0))
            new_img.paste(img_pil, ((bg_width - new_width) // 2, (bg_height - new_height) // 2))
            img_pil = new_img

        elif img_width < bg_width or img_height < bg_height:
            new_img = Image.new("RGBA", (bg_width, bg_height), (0, 0, 0, 0))
            if align == "center":
                new_img.paste(img_pil, ((bg_width - img_width) // 2, (bg_height - img_height) // 2))
            elif align == "top left":
                new_img.paste(img_pil, (0, 0))
            img_pil = new_img
        else:
            if align == "center":
                left = (img_width - bg_width) // 2
                top = (img_height - bg_height) // 2
            elif align == "top left":
                left = 0
                top = 0
            img_pil = img_pil.crop((left, top, left + bg_width, top + bg_height))

        # 确保前景图与背景图尺寸完全一致
        if img_pil.size != image_bg_pil.size:
            img_pil = img_pil.resize(image_bg_pil.size, Image.LANCZOS)
            # print(f"前景图像尺寸已手动调整为背景图像尺寸: {img_pil.size}") # 调试

        # 调整透明度
        alpha_channel = img_pil.split()[-1]
        alpha_channel = Image.eval(alpha_channel, lambda a: int(a * opacity))
        img_pil.putalpha(alpha_channel)

        # 合成图像
        output_image_pil = Image.alpha_composite(image_bg_pil, img_pil)

        return output_image_pil

        
class RN_MultipleImageBlend_2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),  
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),  # 不透明度
                "align": (["size", "top left", "center"],),  # 添加 align 参数
            },
            "optional": {
                "image_b": ("IMAGE",), 
                "image_c": ("IMAGE",), 
                "image_d": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "multiple_image_blend"

    CATEGORY = "Replenish/Image"

    def multiple_image_blend(self, image_a, opacity, align, image_b=None, image_c=None, image_d=None):
        # 创建 images 列表并添加非空的图像
        images = [image_a]
        if image_b is not None:
            images.append(image_b)
        if image_c is not None:
            images.append(image_c)
        if image_d is not None:
            images.append(image_d)

        print("Input image number:", len(images))
        max_height = max(image.shape[1] for image in images)
        max_width = max(image.shape[2] for image in images)
        print(f"Max width: {max_width}; Max height: {max_height}")
        
        image_bg_pil = Image.new("RGBA", (max_width, max_height), (255, 0, 0, 0)) # 创建一个透明背景的输出图像

        blender = RN_ImageBlendBG() # 实例化 RN_ImageBlendBG

        for image in images:
            img_pil = tensor2pil(image).convert("RGBA")

            image_bg_pil = blender.image_blend(image_bg_pil, img_pil, align, opacity) # 使用实例调用 image_blend 方法进行批处理

        output_tensor = pil2tensor(image_bg_pil)
        return (output_tensor,)
    
# 这个基本功能完成了，需要让它更灵活，还要做成批处理 完成
class RN_FillAlpha:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fill_color": ("STRING", {"default": "0,0,0"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fill_alpha"
    CATEGORY = "Replenish/Image"

    def fill_alpha(self, images, fill_color):
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

            image = tensor2pil(image).convert("RGBA")

            output_image = Image.alpha_composite(bg_image, image).convert("RGB") # 合成并转换为RGB模式

            # 转换为张量并存储
            output_tensor = pil2tensor(output_image)
            output_tensors.append(output_tensor)

        # 返回批量图像张量
        batch_output_tensor = torch.cat(output_tensors, dim=0)
        return (batch_output_tensor,)

class RN_ToRGB:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "to_grb"
    CATEGORY = "Replenish/Image"

    def to_grb(self, images):
        output_tensors = []
        for image in images:
            image = tensor2pil(image).convert("RGB")
            output_tensor = pil2tensor(image)
            output_tensors.append(output_tensor)
        batch_output_tensor = torch.cat(output_tensors, dim=0) # 返回批量图像张量
        return (batch_output_tensor,)
    
# 遮罩色阶调节
class RN_MaskLevelsAdjust:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",), 
                "output_black_point": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "output_white_point": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",) 
    FUNCTION = 'levels'
    CATEGORY = "Replenish/Masks"

    def levels(self, masks, output_black_point, output_white_point):
        l_masks = []
        ret_masks = []

        for m in masks:
            l_masks.append(torch.unsqueeze(m, 0))

        for i in range(len(l_masks)):
            _mask = l_masks[i]
            mask_np = np.clip(255. * _mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            orig_mask = Image.fromarray(mask_np, mode="L")

            # 直接对遮罩应用色阶调整
            ret_mask = adjust_levels(orig_mask, 0, 255, 1, output_black_point, output_white_point)
            region_tensor = pil2mask(ret_mask).unsqueeze(0)  # 添加批次维度
            ret_masks.append(region_tensor)

        result_tensor = torch.cat(ret_masks, dim=0) 
        print("Number of masks processed:", len(ret_masks))
        # print("自己遮罩张量的形状:", result_tensor.shape)  # 调试

        return (result_tensor,)
    
class RN_PreviewImageLow:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "quality": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # 不透明度
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True
    CATEGORY = "Replenish/Image"
    DESCRIPTION = "Convert the input image to low-quality JPEG for preview to reduce network pressure."

    def save_images(self, images, quality, filename_prefix="ComfyUI"): # , prompt=None, extra_pnginfo=None
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"  # 保存为 JPG 格式
            quality_int = int(quality * 100)
            img.save(os.path.join(full_output_folder, file), format="JPEG", quality=quality_int)
    
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
    
        return {"ui": {"images": results}}

    
NODE_CLASS_MAPPINGS = {
    "Batch Image Blend": RN_BatchImageBlend,
    "Multiple Image Blend": RN_MultipleImageBlend,
    "Image Blend BG": RN_ImageBlendBG,
    "Fill Alpha": RN_FillAlpha,
    "Mask Levels Adjust": RN_MaskLevelsAdjust,
    "Multiple Image Blend 2": RN_MultipleImageBlend_2,
    "Preview Image-JPEG": RN_PreviewImageLow,
    "To RGB": RN_ToRGB,
    "Reference Resize": RN_Reference_Resize,
    "RN_BatchImageAlign": RN_BatchImageAlign,
}