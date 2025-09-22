import os
from PIL import Image, ImageOps
import numpy as np

def png_to_jpg(image, quality=95):
    """
    将PNG图像无损转换为JPG格式
    如果PNG有透明通道，用白色背景填充
    """
    if image.mode in ('RGBA', 'LA', 'P'):
        # 创建白色背景
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            # 处理调色板模式
            image = image.convert('RGBA')
        # 合成到白色背景上
        background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
        return background
    elif image.mode != 'RGB':
        # 转换其他模式到RGB
        return image.convert('RGB')
    else:
        return image

def resize_with_padding(image, target_width, target_height, pad_color=(0, 0, 0)):
    """
    调整图像尺寸，保持纵横比，用填充补齐目标尺寸
    """
    # 计算缩放比例，保持纵横比
    width_ratio = target_width / image.width
    height_ratio = target_height / image.height
    scale_ratio = min(width_ratio, height_ratio)

    # 计算新尺寸
    new_width = int(image.width * scale_ratio)
    new_height = int(image.height * scale_ratio)

    # 高质量缩放
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # 创建目标尺寸的画布
    new_image = Image.new(image.mode, (target_width, target_height), pad_color)

    # 计算居中位置
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # 将缩放后的图像粘贴到画布中央
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

def center_crop(image, target_width, target_height):
    """
    中心裁剪到目标尺寸
    """
    width, height = image.size

    # 计算裁剪区域
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return image.crop((left, top, right, bottom))

def smart_resize(image, target_width, target_height, method='padding'):
    """
    智能调整图像尺寸
    method: 'padding' - 保持比例+填充, 'crop' - 中心裁剪, 'stretch' - 拉伸
    """
    if method == 'padding':
        return resize_with_padding(image, target_width, target_height)
    elif method == 'crop':
        # 先等比例缩放到至少一边达到目标尺寸
        width_ratio = target_width / image.width
        height_ratio = target_height / image.height
        scale_ratio = max(width_ratio, height_ratio)

        new_width = int(image.width * scale_ratio)
        new_height = int(image.height * scale_ratio)

        # 缩放
        resized = image.resize((new_width, new_height), Image.LANCZOS)

        # 裁剪到目标尺寸
        return center_crop(resized, target_width, target_height)
    elif method == 'stretch':
        return image.resize((target_width, target_height), Image.LANCZOS)
    else:
        raise ValueError("Method must be 'padding', 'crop', or 'stretch'")

def list_available_images(folder_path):
    """列出文件夹中的所有图像文件"""
    if not os.path.exists(folder_path):
        return []

    image_extensions = {'.jpg', '.jpeg',
                        '.png', '.bmp', '.tiff', '.tif', '.gif'}
    images = []

    for file in os.listdir(folder_path):
        if os.path.splitext(file)[1].lower() in image_extensions:
            images.append(file)

    return sorted(images)

def main():
    # 配置参数 - 在这里修改你的设置
    input_folder = "./tests/test_metrics/Image/Process/Input"  # 输入文件夹
    output_folder = "./tests/test_metrics/Image/Process/Output"  # 输出文件夹
    image_name = "1.png"                  # 要处理的图片名
    target_width = 640                    # 目标宽度
    target_height = 427                   # 目标高度
    resize_method = 'padding'             # 调整方法: 'padding', 'crop', 'stretch'
    pad_color = (0, 0, 0)                # 填充颜色 (R, G, B) for padding method
    convert_png_to_jpg = True             # 是否将PNG转换为JPG

    # 处理路径中的反斜杠问题
    input_path = f"{input_folder}"+f"/{image_name}"
    
    # 如果需要转换PNG为JPG，修改输出文件扩展名
    if convert_png_to_jpg and image_name.lower().endswith('.png'):
        output_name = os.path.splitext(image_name)[0] + '.jpg'
    else:
        output_name = image_name
    
    output_path = f"{output_folder}"+f"/{output_name}"

    print(f"Processing image: {image_name}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Target size: {target_width}x{target_height}")
    print(f"Resize method: {resize_method}")
    if convert_png_to_jpg and image_name.lower().endswith('.png'):
        print("PNG to JPG conversion: Enabled")
    print()

    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        print("Please create the folder and put your images there.")
        return

    # 列出可用的图像文件
    available_images = list_available_images(input_folder)
    if available_images:
        print(f"Available images in {input_folder}:")
        for i, img in enumerate(available_images, 1):
            print(f"  {i}. {img}")
        print()
    else:
        print(f"No images found in {input_folder}")
        return

    # 检查指定文件是否存在
    if not os.path.isfile(input_path):
        print(f"Error: Input image not found: {input_path}")
        if available_images:
            print(f"Available images: {', '.join(available_images)}")
            print("Please check the filename and try again.")
        return

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    try:
        # 读取图像
        with Image.open(input_path) as image:
            print(f"Original size: {image.width}x{image.height}")
            print(f"Original mode: {image.mode}")

            # PNG转JPG处理
            if convert_png_to_jpg and image_name.lower().endswith('.png'):
                image = png_to_jpg(image)
                print("Converted PNG to JPG format (transparency removed with white background)")
            elif image.mode not in ('RGB', 'RGBA'):
                # 转换为RGB模式（如果需要）
                image = image.convert('RGB')
                print("Converted to RGB mode")

            # 调整尺寸
            processed_image = smart_resize(
                image, target_width, target_height, resize_method)
            print(
                f"Processed size: {processed_image.width}x{processed_image.height}")

            # 保存图像
            if convert_png_to_jpg and image_name.lower().endswith('.png'):
                # PNG转JPG，使用高质量JPEG格式保存
                processed_image.save(output_path, 'JPEG', quality=95, optimize=True)
            else:
                # 根据原文件扩展名选择保存格式
                file_ext = os.path.splitext(image_name)[1].lower()
                if file_ext in ['.jpg', '.jpeg']:
                    processed_image.save(output_path, 'JPEG', quality=95)
                elif file_ext == '.png':
                    processed_image.save(output_path, 'PNG')
                elif file_ext in ['.bmp', '.tiff', '.tif']:
                    processed_image.save(output_path)
                else:
                    # 默认保存为PNG格式
                    output_path = os.path.splitext(output_path)[0] + '.png'
                    processed_image.save(output_path, 'PNG')

            print(f"Successfully processed and saved to: {output_path}")

            # 显示处理信息
            if resize_method == 'padding':
                print(
                    "Used padding method - aspect ratio preserved, borders added if needed")
            elif resize_method == 'crop':
                print("Used crop method - aspect ratio preserved, image cropped to fit")
            elif resize_method == 'stretch':
                print("Used stretch method - image stretched to exact dimensions")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
