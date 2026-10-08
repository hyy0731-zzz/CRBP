import os
import sys
import cv2
import numpy as np
import paddle
import paddleseg
from paddleseg.models.backbones import STDC1
import torch
import torchvision.transforms as transforms
from PIL import Image

# 添加Matting路径
sys.path.append("Matting")
from ppmatting.models.ppmattingv2 import PPMattingV2

# ===== 在这里设置你的输入图片路径 =====
# 方式1: 处理单张图片
INPUT_IMAGE_PATH = r'D:\hyy\PHOTO\MY-CelebA\original\stargan\original_4.jpg'  # 你的图片路径

# 方式2: 批量处理文件夹中的所有图片 (如果使用这个，上面的单张图片路径会被忽略)
INPUT_IMAGE_DIR = ""  # 替换为图片文件夹路径，例如: "C:/Users/Desktop/photos"

# 输出设置
OUTPUT_DIR = "./matting_output"  # 结果保存目录
THRESHOLD = 0.2  # 掩码阈值 (0.1-0.5之间，数值越大掩码越严格)


# 使用说明:
# 1. 只需要修改上面的 INPUT_IMAGE_PATH 为你的图片路径
# 2. 直接运行: python generate_matting_mask.py
# 3. 结果会保存在 OUTPUT_DIR 目录中
# ===============================================


class MattingMaskGenerator:
    def __init__(self, model_path=None, device='gpu'):
        """
        初始化Matting掩码生成器

        Args:
            model_path: 预训练模型路径
            device: 使用设备 ('gpu' 或 'cpu')
        """
        self.device = device

        # 默认模型路径
        if model_path is None:
            model_path = r'Matting\pretrained_models\ppmattingv2-stdc1-human_512.pdparams'

        # === 加载PaddlePaddle的ppmattingv2模型 ===
        backbone = STDC1(pretrained=None)
        self.matting_model = PPMattingV2(backbone=backbone)
        self.matting_model.eval()

        # 加载预训练权重
        if os.path.exists(model_path):
            self.matting_model.set_state_dict(paddle.load(model_path))
            print(f"✓ 成功加载模型: {model_path}")
        else:
            print(f"✗ 模型文件不存在: {model_path}")
            print("请先下载预训练模型:")
            print(
                "wget https://paddleseg.bj.bcebos.com/matting/models/ppmattingv2-stdc1-human_512.pdparams -P Matting/pretrained_models/")
            sys.exit(1)

        self.matting_model.to(device)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_face_mask(self, img_tensor, threshold=0.2):
        """
        获取人脸掩码 (参考你的代码风格)

        Args:
            img_tensor: torch tensor, [B, 3, H, W], [-1, 1] 或 [0, 1]
            threshold: 阈值，用于生成二值掩码

        Returns:
            mask: torch tensor, [B, 1, H, W], 二值掩码
        """
        # 确保输入在[0,1]范围内
        if img_tensor.min() < 0:
            img_tensor_01 = (img_tensor.detach().cpu() + 1) / 2  # [-1,1] -> [0,1]
        else:
            img_tensor_01 = img_tensor.detach().cpu()  # 已经是[0,1]

        # 转换为numpy并调整维度顺序
        img_np = img_tensor_01.numpy()

        # 转换为paddle tensor
        img_pd = paddle.to_tensor(img_np, dtype='float32')

        # 推理获取alpha掩码
        with paddle.no_grad():
            alpha = self.matting_model({'img': img_pd})  # [B, 1, H, W]

        # 生成二值掩码
        mask = (alpha > threshold).astype('float32')
        mask_np = mask.cpu().numpy()
        mask_torch = torch.from_numpy(mask_np)

        return mask_torch, alpha.cpu().numpy()

    def process_single_image(self, image_path, save_dir='./output', threshold=0.2):
        """
        处理单张图像

        Args:
            image_path: 输入图像路径
            save_dir: 保存目录
            threshold: 掩码阈值
        """
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        # 预处理
        img_tensor = self.transform(image).unsqueeze(0)  # [1, 3, 512, 512]

        # 获取掩码
        binary_mask, alpha_mask = self.get_face_mask(img_tensor, threshold)

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 获取文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 保存结果
        self._save_results(
            original_image=image,
            binary_mask=binary_mask[0, 0],  # [H, W]
            alpha_mask=alpha_mask[0, 0],  # [H, W]
            save_dir=save_dir,
            base_name=base_name,
            original_size=original_size
        )

        print(f"✓ 处理完成: {image_path}")
        print(f"  结果保存在: {save_dir}")

    def _save_results(self, original_image, binary_mask, alpha_mask, save_dir, base_name, original_size):
        """保存处理结果"""

        # 将掩码调整回原始尺寸
        binary_mask_resized = cv2.resize(binary_mask.numpy(), original_size, interpolation=cv2.INTER_CUBIC)
        alpha_mask_resized = cv2.resize(alpha_mask, original_size, interpolation=cv2.INTER_CUBIC)

        # 1. 保存二值掩码 (黑白图)
        binary_mask_255 = (binary_mask_resized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{base_name}_binary_mask.png'), binary_mask_255)

        # 2. 保存Alpha掩码 (灰度图)
        alpha_mask_255 = (alpha_mask_resized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{base_name}_alpha_mask.png'), alpha_mask_255)

        # 3. 保存前景抠图
        original_np = np.array(original_image)
        alpha_3channel = np.stack([alpha_mask_resized] * 3, axis=-1)
        foreground = original_np * alpha_3channel
        cv2.imwrite(os.path.join(save_dir, f'{base_name}_foreground.png'),
                    cv2.cvtColor(foreground.astype(np.uint8), cv2.COLOR_RGB2BGR))

        # 4. 保存可视化对比图
        self._create_visualization(original_image, binary_mask_255, alpha_mask_255,
                                   save_dir, base_name)

    def _create_visualization(self, original_image, binary_mask, alpha_mask, save_dir, base_name):
        """创建可视化对比图"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原图
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 二值掩码
        axes[1].imshow(binary_mask, cmap='gray')
        axes[1].set_title('Binary Mask')
        axes[1].axis('off')

        # Alpha掩码
        axes[2].imshow(alpha_mask, cmap='gray')
        axes[2].set_title('Alpha Mask')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{base_name}_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def process_batch_images(self, image_dir, save_dir='./output', threshold=0.2):
        """
        批量处理图像

        Args:
            image_dir: 图像目录
            save_dir: 保存目录
            threshold: 掩码阈值
        """
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        # 获取所有图像文件
        image_files = []
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_dir, file))

        if not image_files:
            print(f"在目录 {image_dir} 中未找到图像文件")
            return

        print(f"找到 {len(image_files)} 张图像，开始批量处理...")

        # 批量处理
        for i, image_path in enumerate(image_files, 1):
            print(f"处理进度: {i}/{len(image_files)}")
            try:
                self.process_single_image(image_path, save_dir, threshold)
            except Exception as e:
                print(f"✗ 处理失败 {image_path}: {e}")

        print("✓ 批量处理完成!")


def main():
    print("=== Matting掩码生成器 ===")

    # 检查输入路径
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"✗ 图片文件不存在: {INPUT_IMAGE_PATH}")
        print("请检查代码开头设置的 INPUT_IMAGE_PATH 路径是否正确")
        return

    print(f"正在处理图像: {INPUT_IMAGE_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"掩码阈值: {THRESHOLD}")

    # 创建掩码生成器
    generator = MattingMaskGenerator()

    # 处理图像
    generator.process_single_image(INPUT_IMAGE_PATH, OUTPUT_DIR, THRESHOLD)


if __name__ == '__main__':
    # 直接运行
    main()
