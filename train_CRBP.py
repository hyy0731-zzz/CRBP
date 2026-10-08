import argparse
import copy
import json
import os
from os.path import join
import numpy as np
from tqdm import tqdm
from random import shuffle
import torch
import get_output
from model_data_prepare import prepare
from evaluate_fid import evaluate_multiple_models
from attgan.data import check_attribute_conflict
import torch.nn.functional as TFunction
import setGPU
import paddle
import paddleseg
from paddleseg.models.backbones import STDC1
import sys
sys.path.append("D:/hyy/my_code2/Matting")
from ppmatting.models.ppmattingv2 import PPMattingV2
import torchvision.utils as vutils

do_end_to_end = True
do_feature_ensemble = True
do_grad_ensemble = True

do_output_ensemble = False
do_loss_ensemble = False

batch_evaluate = True


def adaptive_feature_normalize(features):
    """
    自适应特征归一化，使不同模型的特征在相似的尺度上
    """
    if features is None:
        return features

    if isinstance(features, list):
        normalized_features = []
        for feat in features:
            if feat is not None and hasattr(feat, 'shape') and len(feat.shape) == 4:
                # 通道级别的归一化
                feat_normalized = torch.nn.functional.normalize(feat, p=2, dim=1)
                normalized_features.append(feat_normalized)
            else:
                normalized_features.append(feat)
        return normalized_features
    else:
        if hasattr(features, 'shape') and len(features.shape) == 4:
            return torch.nn.functional.normalize(features, p=2, dim=1)
        else:
            return features


def enhanced_decoder_attack(clean_decoder_features, adv_decoder_features, attack_utils, model_type="other"):
    """
    改进的解码器中间特征攻击 - 对所有模型公平且统一的策略
    """
    total_decoder_loss = 0

    # 统一的攻击强度，对所有模型公平
    attack_strength = 1.0

    # 特征归一化，使不同模型在相似尺度上
    clean_decoder_features = adaptive_feature_normalize(clean_decoder_features)
    adv_decoder_features = adaptive_feature_normalize(adv_decoder_features)

    if isinstance(clean_decoder_features, list) and isinstance(adv_decoder_features, list):
        # 多层特征的情况
        total_layers = len(clean_decoder_features)

        for layer_idx, (clean_feat, adv_feat) in enumerate(zip(clean_decoder_features, adv_decoder_features)):
            if clean_feat.shape == adv_feat.shape:
                # 1. 基础的特征差异损失 (Wasserstein距离)
                wasserstein_loss = -attack_utils.wasserstein_loss(torch.sum(adv_feat, 1), torch.sum(clean_feat, 1))

                # 2. 结构相似性损失 (适用于所有模型)
                structure_loss = torch.mean((adv_feat - clean_feat) ** 2)

                # 3. 特征分布损失 (JS散度近似)
                mean_clean = torch.mean(clean_feat, dim=(2, 3), keepdim=True)
                mean_adv = torch.mean(adv_feat, dim=(2, 3), keepdim=True)
                distribution_loss = torch.mean((mean_adv - mean_clean) ** 2)

                # 4. 统一的层级权重策略 (适用于所有模型)
                # 中间层权重更高，浅层和深层权重适中
                if total_layers > 1:
                    if layer_idx < total_layers // 3:
                        layer_weight = 0.8  # 浅层
                    elif layer_idx < 2 * total_layers // 3:
                        layer_weight = 1.2  # 中间层 (最重要)
                    else:
                        layer_weight = 1.0  # 深层
                else:
                    layer_weight = 1.0

                # 5. 组合损失
                combined_loss = (wasserstein_loss +
                                 0.3 * structure_loss +
                                 0.2 * distribution_loss)

                total_decoder_loss += layer_weight * combined_loss * attack_strength

    elif clean_decoder_features is not None and adv_decoder_features is not None:
        # 单一张量的情况
        if clean_decoder_features.shape == adv_decoder_features.shape:
            # 使用相同的多重损失策略
            wasserstein_loss = -attack_utils.wasserstein_loss(
                torch.sum(adv_decoder_features, 1),
                torch.sum(clean_decoder_features, 1)
            )

            structure_loss = torch.mean((adv_decoder_features - clean_decoder_features) ** 2)

            mean_clean = torch.mean(clean_decoder_features, dim=(2, 3), keepdim=True)
            mean_adv = torch.mean(adv_decoder_features, dim=(2, 3), keepdim=True)
            distribution_loss = torch.mean((mean_adv - mean_clean) ** 2)

            total_decoder_loss = (wasserstein_loss +
                                  0.3 * structure_loss +
                                  0.2 * distribution_loss) * attack_strength

    return total_decoder_loss


def parse(args=None):
    with open(join('./setting.json'), 'r') as f:
        args_attack = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
    return args_attack


# Init the attacker
def init_get_outputs(args_attack):
    get_output_models = get_output.get_all_features(model=None,
                                                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                                    epsilon=args_attack.attacks.epsilon, args=args_attack.attacks)
    return get_output_models


def just_mean(d_grads):
    d_grads = torch.stack([d for d in d_grads])
    return torch.mean(d_grads, dim=0)

def pcgrad_aggregate(d_grads, eps: float = 1e-12):
    if len(d_grads) == 0:
        raise ValueError('pcgrad_aggregate received empty gradient list')
    grads = [g.clone() for g in d_grads]
    num = len(grads)
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            # Flatten compute inner products across all axes to a scalar
            gij = torch.sum(grads[i] * grads[j])
            if gij < 0:
                denom = torch.sum(grads[j] * grads[j]) + eps
                proj = gij / denom
                grads[i] = grads[i] - proj * grads[j]
    return torch.mean(torch.stack(grads, dim=0), dim=0)

def perform_AttGAN(attack, input_imgs, original_imgs, attibutes, attgan_model, attgan_parse, rand_param):
    att_b_list = [attibutes]
    ## No need to attack all attributes
    # for i in range(attgan_parse.n_attrs):
    #     tmp = attibutes.clone()
    #     tmp[:, i] = 1 - tmp[:, i]
    #     tmp = check_attribute_conflict(tmp, attgan_parse.attrs[i], attgan_parse.attrs)
    #     att_b_list.append(tmp)
    # att_b_list = [att_b_list[0]]
    for i, att_b in enumerate(att_b_list):
        att_b_ = (att_b * 2 - 1) * attgan_parse.thres_int
        if i > 0:
            att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_parse.test_int / attgan_parse.thres_int
        with torch.no_grad():
            gen_noattack, no_attack_middle = attgan_model.G(original_imgs, att_b_)
            # 获取无攻击时的解码器特征
            no_attack_decoder = attgan_model.G.get_decoder_features(original_imgs, att_b_, rand_param)
        adv_gen, adv_gen_middle, adv_decoder_features = attack.get_attgan_features(input_imgs, att_b_, attgan_model,
                                                                                   rand_param)

    return [gen_noattack, adv_gen], [no_attack_middle[-1], adv_gen_middle[-1]], [no_attack_decoder,
                                                                                 adv_decoder_features]


def perform_HiSD(attack, input_imgs, original_imgs, reference_img, E_model, F_model, T_model, G_model, EFTG_models,
                 rand_param):
    # 双向攻击逻辑 - 对HiSD使用双向攻击
    with torch.no_grad():
        # get the original deepfake images
        c = E_model(original_imgs)
        c_trg = c
        s_trg = F_model(reference_img, 1)
        c_trg = T_model(c_trg, s_trg, 1)
        try:
            x_trg, clean_decoder = G_model(c_trg, return_features=True)
        except:
            x_trg = G_model(c_trg)
            clean_decoder = EFTG_models.get_decoder_features(c_trg, rand_param)

    adv_x_trg, adv_c, adv_decoder = attack.get_hisd_features(input_imgs.cuda(), reference_img, F_model, T_model,
                                                             G_model, E_model, EFTG_models, rand_param)

    # Ensure both middle features have the same shape
    if c.shape != adv_c.shape:
        # Try to reshape to match batch size
        if c.shape[0] != adv_c.shape[0]:
            # If batch sizes don't match, use the clean tensor's batch size
            try:
                if len(adv_c.shape) == 3:  # If it's a 3D tensor
                    # Reshape to match clean tensor's batch size
                    channels = adv_c.shape[0] // c.shape[0] if adv_c.shape[0] % c.shape[0] == 0 else adv_c.shape[0]
                    adv_c = adv_c.reshape(c.shape[0], channels, adv_c.shape[1], adv_c.shape[2])
            except RuntimeError:
                # If reshape fails, create a compatible tensor
                adv_c = torch.zeros_like(c)

    return [x_trg, adv_x_trg], [c, adv_c], [clean_decoder, adv_decoder]


def select_model_to_get_feature_pairs(case, img, ori_imgs, reference, attribute_c, attribute_attgan, attack, stargan_s,
                                      atggan_s,
                                      attgan_s, attgan_args, EE, FF, TT, GG, g_models, reconstruct=128, attr_aug=False):
    if attr_aug:
        rand_q = np.random.rand()
    else:
        rand_q = 0
    if case == 0:
        # print('attacking stargan...')
        output_pair, middle_pair, decoder_pair = stargan_s.perform_stargan(img, ori_imgs, attribute_c, attack, rand_q)
    elif case == 1:
        # print('attacking attentiongan...')
        output_pair, middle_pair, decoder_pair = atggan_s.perform_attentiongan(img, ori_imgs, attribute_c, attack,
                                                                               rand_q)
    elif case == 2:
        # print('attacking AttGan...')
        output_pair, middle_pair, decoder_pair = perform_AttGAN(attack, img, ori_imgs, attribute_attgan, attgan_s,
                                                                attgan_args, rand_q)
    elif case == 3:
        # print('attacking HiSD...')
        output_pair, middle_pair, decoder_pair = perform_HiSD(attack, img, ori_imgs, reference, EE, FF, TT, GG,
                                                              g_models, rand_q)
    else:
        raise NotImplementedError('wrong code!')

    # 改进的中间特征处理
    new_middle_pair = []
    for middle in middle_pair:
        if hasattr(middle, 'shape') and len(middle.shape) == 4:
            # 直接插值到目标大小
            new_middle = torch.nn.functional.interpolate(middle, (reconstruct, reconstruct), mode='bilinear')
            new_middle_pair.append(new_middle)
        else:
            # 创建默认张量
            new_middle = torch.zeros(8, 256, reconstruct, reconstruct, device='cuda')
            new_middle_pair.append(new_middle)

    # 改进的解码器特征处理 - 更好地适应不同模型
    new_decoder_pair = []
    for decoder_feat in decoder_pair:
        if isinstance(decoder_feat, list) and len(decoder_feat) > 0:
            # 对于多层特征，使用多尺度融合策略
            processed_features = []
            for feat in decoder_feat:
                if hasattr(feat, 'shape') and len(feat.shape) == 4:
                    try:
                        # 插值到目标大小
                        resized_feat = torch.nn.functional.interpolate(feat, (reconstruct, reconstruct),
                                                                       mode='bilinear')
                        processed_features.append(resized_feat)
                    except:
                        # 如果插值失败，创建兼容张量
                        default_feat = torch.zeros(8, feat.shape[1] if len(feat.shape) > 1 else 128, reconstruct,
                                                   reconstruct, device='cuda')
                        processed_features.append(default_feat)

            # 如果有多个特征，取前3个进行融合（避免过多特征导致内存问题）
            if len(processed_features) > 3:
                processed_features = processed_features[:3]

            # 将多层特征保存为列表，而不是只取第一个
            new_decoder_pair.append(processed_features if len(processed_features) > 1 else processed_features[
                0] if processed_features else None)

        elif decoder_feat is not None and hasattr(decoder_feat, 'shape') and len(decoder_feat.shape) == 4:
            # 单一特征的情况
            try:
                new_decoder = torch.nn.functional.interpolate(decoder_feat, (reconstruct, reconstruct), mode='bilinear')
                new_decoder_pair.append(new_decoder)
            except:
                new_decoder_pair.append(
                    torch.zeros(8, decoder_feat.shape[1] if len(decoder_feat.shape) > 1 else 128, reconstruct,
                                reconstruct, device='cuda'))
        else:
            # 默认情况
            new_decoder_pair.append(torch.zeros(8, 128, reconstruct, reconstruct, device='cuda'))

    return output_pair, new_middle_pair, new_decoder_pair


def DI(X_in):
    import torch.nn.functional as F

    rnd = np.random.randint(256, 290, size=1)[0]
    h_rem = 290 - rnd
    w_rem = 290 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.5:
        X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_right, pad_top, pad_bottom), mode='constant',
                      value=0)
        return F.interpolate(X_out, (256, 256))
    else:
        return F.interpolate(X_in, (256, 256))

def get_face_mask(img_a, matting_model):
    # img_a: torch tensor, [B, 3, H, W], [-1, 1]
    img_a_01 = (img_a.detach().cpu() + 1) / 2  # [0, 1]
    img_np = img_a_01.numpy()
    img_pd = paddle.to_tensor(img_np)
    with paddle.no_grad():
        alpha = matting_model({'img': img_pd})  # [B, 1, H, W]
    mask = (alpha > 0.2).astype('float32')
    mask = mask.cpu().numpy()
    mask = torch.from_numpy(mask).to(img_a.device)
    return mask

def train_attacker():
    args_attack = parse()
    print(args_attack)

    # Init the attacker
    attack_utils = init_get_outputs(args_attack)
    # Init the attacked models
    attack_dataloader, test_dataloader, attgan, attgan_args, stargan_solver, attentiongan_solver, transform, F, T, G, E, reference, gen_models = prepare()

    # === 加载PaddlePaddle的ppmattingv2模型 ===
    backbone = STDC1(pretrained=None)
    matting_model = PPMattingV2(backbone=backbone)
    matting_model.eval()
    matting_model.set_state_dict(
        paddle.load(r'D:\hyy\my_code2\Matting\pretrained_models\ppmattingv2-stdc1-human_512.pdparams'))
    matting_model.to('gpu')

    model_cases = [0, 1, 2, 3]
    import time
    start_time = time.time()

    # Some hyperparameters
    attack_utils.epsilon = 0.05
    reconstruct_feature_size = 32
    iteration_out = 50
    iteration_in = 5
    alpha = 1e-3

    lambda_decoder = 1.0  # 解码器特征攻击的权重，数值高代表解码器权重高，数值低代表编码器权重高

    # pgd
    attack_utils.up = attack_utils.up + torch.tensor(
        np.random.uniform(-attack_utils.epsilon, attack_utils.epsilon, attack_utils.up.shape).astype('float32')).to(
        attack_utils.device)
    momentum = 0

    for t in range(iteration_out):
        print('%dth iter' % t)

        for idx, (img_a, att_a, c_org) in enumerate(tqdm(attack_dataloader)):
            if args_attack.global_settings.num_test is not None and idx * args_attack.global_settings.batch_size == args_attack.global_settings.num_test:
                break
            img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
            att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
            att_a = att_a.type(torch.float)

            # === 获取人脸掩码 ===
            mask = get_face_mask(img_a, matting_model)

            # 可视化保存每个iter的前两张mask
            os.makedirs('./mask_vis', exist_ok=True)
            for i in range(min(2, mask.shape[0])):
                vutils.save_image(mask[i], f'./mask_vis/mask_iter{t}_batch{idx}_img{i}.png')

            if do_feature_ensemble:
                # Feature-Ensemble
                for _ in range(iteration_in):
                    attack_utils.up.requires_grad = True
                    new_input = img_a + attack_utils.up * mask
                    middle_pairs = []
                    decoder_pairs = []
                    shuffle(model_cases)
                    for case in model_cases:
                        _, mid_pair, decoder_pair = select_model_to_get_feature_pairs(case, DI(new_input), img_a,
                                                                                      reference, c_org, att_a,
                                                                                      attack_utils, stargan_solver,
                                                                                      attentiongan_solver, attgan,
                                                                                      attgan_args,
                                                                                      E, F, T, G, gen_models,
                                                                                      reconstruct_feature_size)
                        middle_pairs.append(mid_pair)
                        decoder_pairs.append(decoder_pair)

                    clean_middle_from_models = [middle_pairs[p][0] for p in range(len(middle_pairs))]
                    adv_middle_from_models = [middle_pairs[q][1] for q in range(len(middle_pairs))]

                    # 简化的张量连接
                    try:
                        clean_features_cat = torch.cat(clean_middle_from_models, 1)
                        adv_features_cat = torch.cat(adv_middle_from_models, 1)

                        # 计算编码器损失
                        encoder_loss = -attack_utils.wasserstein_loss(torch.sum(adv_features_cat, 1),
                                                                      torch.sum(clean_features_cat, 1))

                        # 计算解码器损失
                        decoder_loss = 0
                        if lambda_decoder > 0:  # 只有当lambda_decoder > 0时，才计算解码器损失
                            for idx, decoder_pair in enumerate(decoder_pairs):
                                clean_dec, adv_dec = decoder_pair[0], decoder_pair[1]
                                if clean_dec is not None and adv_dec is not None:
                                    # 对所有模型使用统一的策略，不再区分模型类型
                                    pair_decoder_loss = enhanced_decoder_attack(clean_dec, adv_dec, attack_utils,
                                                                                "unified")
                                    decoder_loss += pair_decoder_loss

                        # 损失函数: 编码器损失 + lambda_decoder * 解码器损失
                        loss = encoder_loss + lambda_decoder * decoder_loss

                    except RuntimeError:
                        # 简化的错误处理 - 回到基本方法
                        encoder_loss = 0
                        for clean_feat, adv_feat in zip(clean_middle_from_models, adv_middle_from_models):
                            if clean_feat.shape == adv_feat.shape:
                                pair_loss = -attack_utils.wasserstein_loss(torch.sum(adv_feat, 1),
                                                                           torch.sum(clean_feat, 1))
                                encoder_loss += pair_loss

                        # 计算解码器损失
                        decoder_loss = 0
                        if lambda_decoder > 0:  # 只有当lambda_decoder > 0时，才计算解码器损失
                            for idx, decoder_pair in enumerate(decoder_pairs):
                                clean_dec, adv_dec = decoder_pair[0], decoder_pair[1]
                                if clean_dec is not None and adv_dec is not None:
                                    # 对所有模型使用统一的策略
                                    pair_decoder_loss = enhanced_decoder_attack(clean_dec, adv_dec, attack_utils,
                                                                                "unified")
                                    decoder_loss += pair_decoder_loss

                        # 损失函数: 编码器损失 + lambda_decoder * 解码器损失
                        loss = encoder_loss + lambda_decoder * decoder_loss

                        loss.backward()

                        grad_c = attack_utils.up.grad.clone().to(attack_utils.device)
                        grad_c_hat = grad_c / (torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1e-12)
                        attack_utils.up.grad.zero_()

                        attack_utils.up.data = attack_utils.up.data - alpha * torch.sign(grad_c_hat)
                        attack_utils.up.data = attack_utils.up.data.clamp(-attack_utils.epsilon, attack_utils.epsilon)
                        attack_utils.up = attack_utils.up.detach()
                        continue

                    # 计算编码器损失
                    encoder_loss = -attack_utils.wasserstein_loss(torch.sum(adv_features_cat, 1),
                                                                  torch.sum(clean_features_cat, 1))

                    # 计算解码器损失
                    decoder_loss = 0
                    if lambda_decoder > 0:  # 只有当lambda_decoder > 0时，才计算解码器损失
                        for idx, decoder_pair in enumerate(decoder_pairs):
                            clean_dec, adv_dec = decoder_pair[0], decoder_pair[1]
                            if clean_dec is not None and adv_dec is not None:
                                # 对所有模型使用统一的策略
                                pair_decoder_loss = enhanced_decoder_attack(clean_dec, adv_dec, attack_utils, "unified")
                                decoder_loss += pair_decoder_loss

                    # 损失函数: 编码器损失 + lambda_decoder * 解码器损失
                    loss = encoder_loss + lambda_decoder * decoder_loss
                    loss.backward()

                    grad_c = attack_utils.up.grad.clone().to(attack_utils.device)
                    grad_c_hat = grad_c / (torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1e-12)
                    attack_utils.up.grad.zero_()

                    attack_utils.up.data = attack_utils.up.data - alpha * torch.sign(grad_c_hat)
                    attack_utils.up.data = attack_utils.up.data.clamp(-attack_utils.epsilon, attack_utils.epsilon)
                    attack_utils.up = attack_utils.up.detach()

            if do_end_to_end:
                # End-to-End Ensemble
                attack_utils.up.requires_grad = True
                new_new_input = img_a + attack_utils.up * mask
                output_grads = []  # grad_ensemble
                logit_pairs = []  # logit_ensemble
                out_loss = 0  # loss_ensemble

                for case in model_cases:
                    out_pair, _, decoder_pair = select_model_to_get_feature_pairs(case, DI(new_new_input), img_a,
                                                                                  reference, c_org, att_a,
                                                                                  attack_utils, stargan_solver,
                                                                                  attentiongan_solver, attgan,
                                                                                  attgan_args,
                                                                                  E, F, T, G, gen_models,
                                                                                  reconstruct_feature_size)

                    # 计算输出图像损失
                    loss_one = -1 * attack_utils.loss_fn(out_pair[0], out_pair[1])

                    if do_grad_ensemble:
                        loss_one.backward(retain_graph=True)
                        grad_case = attack_utils.up.grad.clone().to(attack_utils.device)
                        grad_case = grad_case / (torch.mean(torch.abs(grad_case), (1, 2, 3), keepdim=True) + 1e-12)
                        output_grads.append(grad_case)
                        attack_utils.up.grad.zero_()
                    elif do_loss_ensemble:
                        out_loss += loss_one
                    elif do_output_ensemble:
                        logit_pairs.append(out_pair)

                if do_grad_ensemble:
                    grad_cout_hat = pcgrad_aggregate(output_grads)
                elif do_loss_ensemble:
                    out_loss.backward()
                    grad_cout_hat = attack_utils.up.grad.clone().to(attack_utils.device)
                    grad_cout_hat = grad_cout_hat / (
                                torch.mean(torch.abs(grad_cout_hat), (1, 2, 3), keepdim=True) + 1e-12)
                elif do_output_ensemble:
                    clean_out = [logit_pairs[p][0] for p in range(len(logit_pairs))]
                    adv_out = [logit_pairs[q][1] for q in range(len(logit_pairs))]

                    # 简化的张量连接
                    try:
                        clean_cat = torch.cat(clean_out, 1)
                        adv_cat = torch.cat(adv_out, 1)
                        logit_loss = -1 * attack_utils.loss_fn(torch.sum(adv_cat, 1), torch.sum(clean_cat, 1))
                        logit_loss.backward()
                        grad_cout_hat = attack_utils.up.grad.clone().to(attack_utils.device)
                        grad_cout_hat = grad_cout_hat / (
                                    torch.mean(torch.abs(grad_cout_hat), (1, 2, 3), keepdim=True) + 1e-12)
                    except RuntimeError:
                        # 如果连接失败，使用简单方法
                        grad_cout_hat = torch.zeros_like(attack_utils.up)
                        for out_pair in logit_pairs:
                            loss_one = -1 * attack_utils.loss_fn(out_pair[0], out_pair[1])
                            loss_one.backward(retain_graph=True)
                            grad_case = attack_utils.up.grad.clone().to(attack_utils.device)
                            grad_case = grad_case / (torch.mean(torch.abs(grad_case), (1, 2, 3), keepdim=True) + 1e-12)
                            grad_cout_hat += grad_case
                            attack_utils.up.grad.zero_()
                        grad_cout_hat = grad_cout_hat / len(logit_pairs)
                else:
                    raise NotImplementedError('choose one ensemble from grad/loss/logit')

                # MI
                grad_cout_hat = grad_cout_hat + 0.8 * momentum
                momentum = grad_cout_hat

                attack_utils.up.data = attack_utils.up.data - alpha * torch.sign(grad_cout_hat)
                attack_utils.up.data = attack_utils.up.data.clamp(-attack_utils.epsilon, attack_utils.epsilon)
                attack_utils.up = attack_utils.up.detach()

        print('up:', torch.max(attack_utils.up), torch.min(attack_utils.up))

        if batch_evaluate and t % 10 == 0:
            _, _, _, _ = evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, stargan_solver,
                                                  attentiongan_solver,
                                                  transform, F, T, G, E, reference, gen_models, attack_utils,
                                                  max_samples=20)

    end_time = time.time()
    print('cost time:', end_time - start_time)
    torch.save(attack_utils.up, 'final.pt')
    _, _, _, _ = evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, stargan_solver,
                                          attentiongan_solver,
                                          transform, F, T, G, E, reference, gen_models, attack_utils,
                                          max_samples=1000)


if __name__ == "__main__":
    train_attacker()
