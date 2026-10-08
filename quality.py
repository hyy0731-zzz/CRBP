#!/usr/bin/env python3
"""
评估三种扰动方法生成的对抗样本的图片质量
测试final.pt, pert_FOUND.pt和perturbation.pt三种方法
使用PSNR, SSIM和LPIPS三种指标
在CelebA测试集上测试50张图片取平均值
"""

import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# 图像质量评估指标
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# LPIPS
try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告: LPIPS 库未安装，将跳过LPIPS计算")
    print("安装命令: pip install lpips")

from model_data_prepare import prepare
from data import CelebA
import matplotlib.pyplot as plt
import seaborn as sns


class ImageQualityEvaluator:
    """图片质量评估器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # 初始化LPIPS模型
        if LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(device)
            print("LPIPS模型初始化成功")
        else:
            self.lpips_model = None

    def tensor_to_numpy(self, tensor):
        """将PyTorch tensor转换为numpy array用于PSNR和SSIM计算"""
        # 假设tensor格式为 [B, C, H, W] 且值在[-1, 1]范围内
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 去掉batch维度

        # 从[-1,1]转换到[0,1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)

        # 转换为numpy并调整维度顺序 [C,H,W] -> [H,W,C]
        numpy_img = tensor.detach().cpu().numpy()
        if numpy_img.ndim == 3:
            numpy_img = np.transpose(numpy_img, (1, 2, 0))

        return numpy_img

    def calculate_psnr(self, img1, img2):
        """计算PSNR"""
        try:
            img1_np = self.tensor_to_numpy(img1)
            img2_np = self.tensor_to_numpy(img2)
            return psnr(img1_np, img2_np, data_range=1.0)
        except Exception as e:
            print(f"PSNR计算错误: {e}")
            return 0.0

    def calculate_ssim(self, img1, img2):
        """计算SSIM"""
        try:
            img1_np = self.tensor_to_numpy(img1)
            img2_np = self.tensor_to_numpy(img2)

            if img1_np.ndim == 3:  # 彩色图像
                return ssim(img1_np, img2_np, multichannel=True, channel_axis=2, data_range=1.0)
            else:  # 灰度图像
                return ssim(img1_np, img2_np, data_range=1.0)
        except Exception as e:
            print(f"SSIM计算错误: {e}")
            return 0.0

    def calculate_lpips(self, img1, img2):
        """计算LPIPS感知距离"""
        if not LPIPS_AVAILABLE or self.lpips_model is None:
            return 0.0

        try:
            # LPIPS期望输入为[-1, 1]范围的tensor
            with torch.no_grad():
                # 确保输入在正确的设备上且格式正确
                if img1.dim() == 3:
                    img1 = img1.unsqueeze(0)
                if img2.dim() == 3:
                    img2 = img2.unsqueeze(0)

                img1 = img1.to(self.device)
                img2 = img2.to(self.device)

                lpips_score = self.lpips_model(img1, img2)
                return lpips_score.item()
        except Exception as e:
            print(f"LPIPS计算错误: {e}")
            return 0.0


class PerturbationLoader:
    """扰动文件加载器"""

    def __init__(self, epsilon=0.05):
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_perturbation(self, file_path):
        """加载扰动文件"""
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在")
            return None

        try:
            perturbation = torch.load(file_path, map_location=self.device)

            # 检查是否为有效的张量
            if not isinstance(perturbation, torch.Tensor):
                print(f"警告: {file_path} 不是有效的torch.Tensor")
                return None

            # 确保至少有4个维度 [B, C, H, W]
            if perturbation.dim() < 4:
                print(f"扩展维度: 从 {perturbation.shape} ", end="")
                while perturbation.dim() < 4:
                    perturbation = perturbation.unsqueeze(0)
                print(f"到 {perturbation.shape}")

            # 限制扰动强度
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)

            print(f"成功加载扰动: {file_path}")
            print(f"  形状: {perturbation.shape}")
            print(f"  数据类型: {perturbation.dtype}")
            print(f"  设备: {perturbation.device}")
            print(f"  范围: [{perturbation.min():.4f}, {perturbation.max():.4f}]")
            print(f"  L2范数: {torch.norm(perturbation).item():.4f}")

            # 检查是否需要通道数调整的警告
            if perturbation.shape[1] not in [1, 3]:
                print(f"  警告: 扰动有 {perturbation.shape[1]} 个通道，可能需要调整")

            return perturbation
        except Exception as e:
            print(f"加载 {file_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_baseline_perturbation(self, shape=(1, 3, 256, 256)):
        """创建基准扰动（用于对比）"""
        print("创建基准零扰动")
        return torch.zeros(shape).to(self.device)


def generate_adversarial_samples(models, clean_images, att_a, c_org, perturbations):
    """生成对抗样本 - 直接比较原始图片和加扰动后的图片"""
    results = {}

    for pert_name, perturbation in perturbations.items():
        if perturbation is None:
            continue

        print(f"使用 {pert_name} 生成对抗样本...")

        # 检查并调整扰动维度
        print(f"  Clean images shape: {clean_images.shape}")
        print(f"  Perturbation shape: {perturbation.shape}")

        # 确保扰动有正确的通道数（应该是3个通道用于RGB图像）
        if perturbation.shape[1] != clean_images.shape[1]:
            print(f"  警告: 扰动通道数 ({perturbation.shape[1]}) 与图像通道数 ({clean_images.shape[1]}) 不匹配")
            if perturbation.shape[1] == 1:
                # 如果扰动是单通道，复制到3个通道
                perturbation = perturbation.repeat(1, 3, 1, 1)
                print(f"  已将单通道扰动复制到3通道")
            elif perturbation.shape[1] == 2:
                # 如果扰动是2通道，添加第3个通道
                third_channel = torch.zeros_like(perturbation[:, :1, :, :])
                perturbation = torch.cat([perturbation, third_channel], dim=1)
                print(f"  已添加第3个通道，新形状: {perturbation.shape}")
            elif perturbation.shape[1] > 3:
                # 如果扰动通道数太多，只取前3个
                perturbation = perturbation[:, :3, :, :]
                print(f"  已截取前3个通道，新形状: {perturbation.shape}")

        # 调整batch大小
        if perturbation.shape[0] == 1 and clean_images.shape[0] > 1:
            # 如果扰动是单张，需要扩展到batch大小
            perturbation = perturbation.repeat(clean_images.shape[0], 1, 1, 1)
        elif perturbation.shape[0] > clean_images.shape[0]:
            # 如果扰动batch太大，截取需要的部分
            perturbation = perturbation[:clean_images.shape[0]]

        # 调整空间维度（高度和宽度）
        if perturbation.shape[2] != clean_images.shape[2] or perturbation.shape[3] != clean_images.shape[3]:
            print(f"  调整扰动空间维度从 {perturbation.shape[2:]} 到 {clean_images.shape[2:]}")
            perturbation = torch.nn.functional.interpolate(
                perturbation, size=(clean_images.shape[2], clean_images.shape[3]),
                mode='bilinear', align_corners=False
            )

        print(f"  最终扰动形状: {perturbation.shape}")

        # 生成对抗样本（原始图片 + 扰动）
        adversarial_images = clean_images + perturbation
        adversarial_images = torch.clamp(adversarial_images, -1, 1)

        # 保存结果 - 直接比较原始图片和对抗样本图片
        results[pert_name] = {
            'clean_images': clean_images,  # 原始图片
            'adversarial_images': adversarial_images,  # 加扰动后的对抗样本
            'perturbation': perturbation  # 扰动本身（用于可视化）
        }

        print(f"  ✓ 成功生成对抗样本")

    return results


def evaluate_single_method(evaluator, results, method_name, num_samples):
    """评估单个方法的图片质量 - 比较原始图片和对抗样本图片"""
    if method_name not in results:
        return None

    method_results = results[method_name]
    clean_images = method_results['clean_images']  # 原始图片
    adversarial_images = method_results['adversarial_images']  # 对抗样本图片

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    print(f"\n评估 {method_name} 方法...")
    print(f"比较原始图片 vs 对抗样本图片")

    for i in tqdm(range(min(num_samples, clean_images.shape[0])), desc="计算指标"):
        clean_img = clean_images[i:i + 1]
        adv_img = adversarial_images[i:i + 1]

        # 计算PSNR（数值越高表示图片质量越好，扰动越小）
        psnr_score = evaluator.calculate_psnr(clean_img, adv_img)
        psnr_scores.append(psnr_score)

        # 计算SSIM（数值越高表示结构相似性越好，扰动对结构影响越小）
        ssim_score = evaluator.calculate_ssim(clean_img, adv_img)
        ssim_scores.append(ssim_score)

        # 计算LPIPS（数值越低表示感知相似性越好，扰动在感知上越不明显）
        lpips_score = evaluator.calculate_lpips(clean_img, adv_img)
        lpips_scores.append(lpips_score)

    return {
        'PSNR': {
            'scores': psnr_scores,
            'mean': np.mean(psnr_scores),
            'std': np.std(psnr_scores)
        },
        'SSIM': {
            'scores': ssim_scores,
            'mean': np.mean(ssim_scores),
            'std': np.std(ssim_scores)
        },
        'LPIPS': {
            'scores': lpips_scores,
            'mean': np.mean(lpips_scores),
            'std': np.std(lpips_scores)
        }
    }


def save_visualization_samples(results, output_dir, num_samples=5):
    """保存可视化样本 - 原始图片、对抗样本图片和扰动"""
    os.makedirs(output_dir, exist_ok=True)

    for method_name, method_results in results.items():
        method_dir = os.path.join(output_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

        clean_images = method_results['clean_images']  # 原始图片
        adversarial_images = method_results['adversarial_images']  # 对抗样本
        perturbations = method_results['perturbation']  # 扰动

        for i in range(min(num_samples, clean_images.shape[0])):
            # 保存原始图片
            clean_img = (clean_images[i] + 1) / 2  # 从[-1,1]转换到[0,1]
            clean_img = torch.clamp(clean_img, 0, 1)

            # 保存对抗样本图片
            adv_img = (adversarial_images[i] + 1) / 2
            adv_img = torch.clamp(adv_img, 0, 1)

            # 保存扰动（放大显示，便于观察）
            pert_img = perturbations[i if i < perturbations.shape[0] else 0]  # 处理batch大小不匹配
            # 将扰动从[-epsilon, epsilon]范围映射到[0, 1]以便可视化
            epsilon = 0.05  # 假设epsilon为0.05
            pert_img = (pert_img + epsilon) / (2 * epsilon)
            pert_img = torch.clamp(pert_img, 0, 1)

            # 使用torchvision保存图片
            import torchvision.utils as vutils
            vutils.save_image(clean_img, os.path.join(method_dir, f'original_{i}.png'))
            vutils.save_image(adv_img, os.path.join(method_dir, f'adversarial_{i}.png'))
            vutils.save_image(pert_img, os.path.join(method_dir, f'perturbation_{i}.png'))


def plot_comparison_charts(evaluation_results, output_dir):
    """绘制对比图表"""
    os.makedirs(output_dir, exist_ok=True)

    methods = list(evaluation_results.keys())
    metrics = ['PSNR', 'SSIM', 'LPIPS']

    # 准备数据
    means = {metric: [] for metric in metrics}
    stds = {metric: [] for metric in metrics}

    for method in methods:
        if evaluation_results[method] is not None:
            for metric in metrics:
                means[metric].append(evaluation_results[method][metric]['mean'])
                stds[metric].append(evaluation_results[method][metric]['std'])
        else:
            for metric in metrics:
                means[metric].append(0)
                stds[metric].append(0)

    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x_pos = np.arange(len(methods))

        bars = ax.bar(x_pos, means[metric], yerr=stds[metric],
                      capsize=5, alpha=0.7, color=['blue', 'orange', 'green'])

        ax.set_xlabel('扰动方法')
        ax.set_ylabel(f'{metric} 分数')
        ax.set_title(f'{metric} 对比 (原始图片 vs 对抗样本)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45)
        ax.grid(True, alpha=0.3)

        # 在柱子上添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 创建详细的数值对比表
    print("\n" + "=" * 60)
    print("对抗样本图片质量评估结果汇总")
    print("(比较原始图片 vs 加扰动后的对抗样本)")
    print("=" * 60)
    print("PSNR: 越高越好 (扰动越小)")
    print("SSIM: 越高越好 (结构保持越好)")
    print("LPIPS: 越低越好 (感知差异越小)")
    print("=" * 60)

    print(f"{'方法':<15} {'PSNR':<12} {'SSIM':<12} {'LPIPS':<12}")
    print("-" * 60)

    for method in methods:
        if evaluation_results[method] is not None:
            psnr_mean = evaluation_results[method]['PSNR']['mean']
            ssim_mean = evaluation_results[method]['SSIM']['mean']
            lpips_mean = evaluation_results[method]['LPIPS']['mean']
            print(f"{method:<15} {psnr_mean:<12.3f} {ssim_mean:<12.3f} {lpips_mean:<12.3f}")
        else:
            print(f"{method:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12}")


def main():
    parser = argparse.ArgumentParser(description='评估三种扰动方法的图片质量')
    parser.add_argument('--num_samples', type=int, default=50, help='测试样本数量')
    parser.add_argument('--output_dir', type=str, default='quality_evaluation_results', help='输出目录')
    parser.add_argument('--epsilon', type=float, default=0.05, help='扰动强度')

    args = parser.parse_args()

    print("开始图片质量评估...")
    print(f"测试样本数: {args.num_samples}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    # 初始化评估器
    evaluator = ImageQualityEvaluator()
    pert_loader = PerturbationLoader(epsilon=args.epsilon)

    # 加载扰动文件
    perturbation_files = {
        'final.pt': './final.pt',
        'pert_FOUND.pt': './pert_FOUND.pt',
        'perturbation.pt': './perturbation.pt'  # 如果不存在会显示警告
    }

    perturbations = {}
    for name, path in perturbation_files.items():
        pert = pert_loader.load_perturbation(path)
        if pert is not None:
            perturbations[name] = pert
        else:
            # 如果是perturbation.pt不存在，创建一个基准扰动
            if name == 'perturbation.pt':
                print("创建基准零扰动作为perturbation.pt的替代")
                perturbations[name] = pert_loader.create_baseline_perturbation()

    if not perturbations:
        print("错误: 没有找到任何有效的扰动文件")
        return

    # 加载模型和数据
    print("\n加载模型和数据...")
    attack_dataloader, test_dataloader, attgan, attgan_args, stargan_solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()

    models = {
        'attgan': attgan,
        'attgan_args': attgan_args,
        'stargan_solver': stargan_solver,
        'attentiongan_solver': attentiongan_solver
    }

    # 限制测试数据集
    test_indices = list(range(min(args.num_samples, len(test_dataloader.dataset))))
    test_subset = Subset(test_dataloader.dataset, test_indices)
    limited_test_loader = DataLoader(
        test_subset,
        batch_size=8,  # 使用较小的batch size
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # 收集所有结果
    all_results = {}
    evaluation_results = {}

    # 逐批处理数据
    for batch_idx, (img_a, att_a, c_org) in enumerate(tqdm(limited_test_loader, desc="处理批次")):
        img_a = img_a.cuda() if torch.cuda.is_available() else img_a
        att_a = att_a.cuda() if torch.cuda.is_available() else att_a
        att_a = att_a.type(torch.float)

        # 生成对抗样本
        batch_results = generate_adversarial_samples(models, img_a, att_a, c_org, perturbations)

        # 合并结果
        for method_name, method_data in batch_results.items():
            if method_name not in all_results:
                all_results[method_name] = {key: [] for key in method_data.keys()}

            for key, value in method_data.items():
                all_results[method_name][key].append(value)

    # 合并所有批次的结果
    for method_name in all_results:
        for key in all_results[method_name]:
            all_results[method_name][key] = torch.cat(all_results[method_name][key], dim=0)

    # 评估每种方法
    for method_name in perturbations.keys():
        evaluation_results[method_name] = evaluate_single_method(
            evaluator, all_results, method_name, args.num_samples
        )

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存可视化样本
    print("\n保存可视化样本...")
    sample_dir = os.path.join(args.output_dir, 'samples')
    save_visualization_samples(all_results, sample_dir)

    # 绘制对比图表
    print("生成对比图表...")
    plot_comparison_charts(evaluation_results, args.output_dir)

    # 保存详细结果
    results_file = os.path.join(args.output_dir, 'detailed_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        json_results = {}
        for method, results in evaluation_results.items():
            if results is not None:
                json_results[method] = {}
                for metric, metric_data in results.items():
                    json_results[method][metric] = {
                        'mean': float(metric_data['mean']),
                        'std': float(metric_data['std']),
                        'scores': [float(x) for x in metric_data['scores']]
                    }
            else:
                json_results[method] = None

        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\n评估完成! 结果已保存到: {args.output_dir}")
    print(f"详细结果: {results_file}")
    print(f"对比图表: {os.path.join(args.output_dir, 'quality_comparison.png')}")
    print(f"样本图片: {sample_dir}")


if __name__ == "__main__":
    main()
