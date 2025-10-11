import os
import math
from PIL import Image
import numpy as np
import csv


def read_image(path):
    """读取图像并转换为numpy数组"""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32)


def psnr(img1, img2, max_val=255.0):
    """
    计算 PSNR
    img1, img2: numpy数组 HxWxC, float32, 取值范围 0..255
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def center_crop_to_min(lq, gt):
    """如果两张图尺寸不一致，中心裁剪到相同的最小尺寸"""
    h = min(lq.shape[0], gt.shape[0])
    w = min(lq.shape[1], gt.shape[1])

    def crop(img, h, w):
        H, W = img.shape[:2]
        y0 = (H - h) // 2
        x0 = (W - w) // 2
        return img[y0:y0+h, x0:x0+w]

    return crop(lq, h, w), crop(gt, h, w)


def main(image_name):

    save_lq_to_gt = False        # 是否将Lq图像保存到Gt文件夹

    lq_path = os.path.join("tests", "test_metrics", "Image", "Lq", image_name)
    gt_path = os.path.join("tests", "test_metrics", "Image", "Gt", image_name)

    print(f"Processing image: {image_name}")
    print(f"LQ path: {lq_path}")
    print(f"GT path: {gt_path}")

    # 检查文件是否存在
    if not os.path.isfile(lq_path):
        print(f"Error: LQ image not found: {lq_path}")
        return
    if not os.path.isfile(gt_path):
        print(f"Error: GT image not found: {gt_path}")
        return

    # 读取图像
    try:
        lq = read_image(lq_path)
        gt = read_image(gt_path)
        print(f"LQ image shape: {lq.shape}")
        print(f"GT image shape: {gt.shape}")
    except Exception as e:
        print(f"Error reading images: {e}")
        return

    # 尺寸不一致时，中心裁剪到相同尺寸
    if lq.shape != gt.shape:
        print("Images have different shapes, center cropping to minimum size...")
        lq, gt = center_crop_to_min(lq, gt)
        print(f"After cropping - LQ shape: {lq.shape}, GT shape: {gt.shape}")

    # 计算PSNR
    psnr_value = psnr(lq, gt)
    print(f"PSNR: {psnr_value:.4f} dB")

    # 可选：保存Lq到Gt文件夹
    if save_lq_to_gt:
        out_dir = os.path.join("Image", "Gt")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, image_name)
        Image.fromarray(np.uint8(lq.clip(0, 255))).save(out_path)
        print(f"Saved Lq copy to: {out_path}")

    # 保存结果到CSV
    with open('psnr_result.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'psnr_db'])
        writer.writerow([image_name, f'{psnr_value:.4f}'])

    print("Results saved to psnr_result.csv")


if __name__ == "__main__":
    image_name = "0047.png"  # 替换为你的图像文件名
    main(image_name)
