import numpy as np
import torch
from PIL import Image
import cv2
import math

def readable_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"

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


def image_blend(image_bg_pil, img_pil, align, opacity):
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

