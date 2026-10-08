#!/usr/bin/env python3
"""
简单测试脚本 - 直接使用现有的evaluate_fid.py测试代码
"""

import torch
import argparse
from model_data_prepare import prepare
from evaluate_fid import evaluate_multiple_models


def load_perturbation(pert_path='pert_FOUND.pt', epsilon=0.05):
    """加载预训练的扰动"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建一个简单的攻击对象来存储扰动
    class SimpleAttack:
        def __init__(self, perturbation, epsilon):
            self.up = perturbation
            self.epsilon = epsilon

    # 加载扰动
    perturbation = torch.load(pert_path, map_location=device)
    perturbation = torch.clamp(perturbation, -epsilon, epsilon)

    print(f"加载扰动文件: {pert_path}")
    print(f"扰动形状: {perturbation.shape}")
    print(f"扰动范围: [{perturbation.min():.4f}, {perturbation.max():.4f}]")

    return SimpleAttack(perturbation, epsilon)


def main():
    parser = argparse.ArgumentParser(description='简单测试脚本')
    parser.add_argument('--perturbation', type=str, default='final.pt',
                        help='扰动文件路径')
    parser.add_argument('--samples', type=int, default=1000,
                        help='测试样本数量')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='扰动强度')
    parser.add_argument('--log_file', type=str, default='test_results.txt',
                        help='结果文件')

    args = parser.parse_args()

    print("=" * 50)
    print("简单测试开始")
    print("=" * 50)
    print(f"扰动文件: {args.perturbation}")
    print(f"测试样本: {args.samples}")
    print(f"扰动强度: {args.epsilon}")
    print(f"结果文件: {args.log_file}")

    # 加载扰动
    attack_obj = load_perturbation(args.perturbation, args.epsilon)

    # 准备模型和数据 (直接用现有的prepare函数)
    print("\n加载模型和数据...")
    from train_FOUND import parse
    args_attack = parse()

    attack_dataloader, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()

    # 直接调用现有的测试函数
    print(f"\n开始测试 {args.samples} 个样本...")
    evaluate_multiple_models(
        args_attack,
        test_dataloader,
        attgan,
        attgan_args,
        solver,
        attentiongan_solver,
        transform,
        F, T, G, E,
        reference,
        gen_models,
        attack_obj,  # 我们的扰动对象
        max_samples=args.samples,
        log_file=args.log_file
    )

    print(f"\n测试完成！结果保存在: {args.log_file}")


if __name__ == "__main__":
    main() 