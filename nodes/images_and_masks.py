import numpy as np
import torch
from PIL import Image
import cv2

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

###################################################

class BatchImageBlend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 接收多批次的图像输入
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),  # 透明度
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_blend"

    CATEGORY = "Replenish/Image"

    def image_blend(self, images, opacity):
        # 找到最大高度和宽度
        max_height = max(image.shape[0] for image in images)
        max_width = max(image.shape[1] for image in images)
        print(f"Max width: {max_width}; Max height:, {max_height}")
        
        # 创建一个空白输出图像
        output_image = Image.new("RGBA", (max_width, max_height), (0, 0, 0, 0))

        for image in images:
            print("input tensor space:", image.shape)
            img = tensor2pil(image)
            img = img.convert("RGBA")  # 确保图像是RGBA模式
            
            # 计算位置，将图像中心对齐
            x_pos = (max_width - img.width) // 2
            y_pos = (max_height - img.height) // 2
            
            # 创建新图像以保存透明度处理后的结果
            img_with_opacity = Image.new("RGBA", img.size)

            for x in range(img.width):
                for y in range(img.height):
                    r, g, b, a = img.getpixel((x, y))
                    new_alpha = int(a * opacity)  # 乘以原始透明度
                    img_with_opacity.putpixel((x, y), (r, g, b, new_alpha))

            # 将图像粘贴到输出图像
            output_image.paste(img_with_opacity, (x_pos, y_pos), img_with_opacity)

        # 打印输出图像的尺寸和张量形状
        output_tensor = pil2tensor(output_image)
        print("Output tensor shape:", output_tensor.shape)  # 输出张量的形状

        return (output_tensor, )


# 遮罩色阶调节
class MaskLevelsAdjust:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),  # 修改为处理遮罩
                "output_black_point": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "output_white_point": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)  # 修改返回名称为“masks”
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
            region_tensor = pil2mask(ret_mask).unsqueeze(0).unsqueeze(1)
            ret_masks.append(region_tensor)
            
        # 将所有处理后的遮罩合并为一个张量
        result_tensor = torch.cat(ret_masks, dim=0)

        # 打印输出信息
        print("Number of masks processed:", len(ret_masks))
        print("Shape of the result tensor:", result_tensor.shape)

        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "Batch Image Blend": BatchImageBlend,
    "Mask Levels Adjust": MaskLevelsAdjust,
}
