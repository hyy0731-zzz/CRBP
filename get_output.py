import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm
import torch.nn.functional as Function
import sklearn.metrics as skm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
import os


class Wasserstein_loss(nn.Module):
    def __init__(self) -> None:
        super(Wasserstein_loss, self).__init__()

    def forward(self, input, target):
        return torch.mean(input * target)


try:
    import defenses.smoothing as smoothing
except:
    import attgan.defenses.smoothing as smoothing


class get_all_features(object):
    def __init__(self, model=None, device='cuda', epsilon=0.05, args=None):
        """
        epsilon: magnitude of attack
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.MSELoss().to(device)
        self.wasserstein_loss = Wasserstein_loss().to(device)
        self.device = device

        self.up = torch.zeros([1, 3, 256, 256]).to(self.device)

    def get_attgan_features(self, X_input, X_att, attgan, rand_param):
        # attribute augmentation, default=No
        q = rand_param
        attr_att = torch.tensor(np.random.uniform(-0.5, 0.5, X_att.size()).astype('float32')).to(self.device)
        new_attr = (X_att * 0.8 + attr_att * 0.2) if q > 0.2 else X_att

        # 保留原有AttGAN逻辑
        output, middle = attgan.G(X_input, new_attr)
        # get_decoder_features返回一个特征列表
        decoder_features = attgan.G.get_decoder_features(X_input, new_attr, rand_param)
        attgan.G.zero_grad()
        return output, middle, decoder_features

    def get_stargan_features(self, X_input, c_trg, model, rand_param):
        # 获取干净特征
        with torch.no_grad():
            clean_output, clean_feats, clean_middle = model.forward_my_attack(X_input, c_trg, rand_param)
            clean_decoder_feats = model.get_decoder_features(X_input, c_trg, rand_param)

        # 获取对抗特征
        adv_output, adv_feats, adv_middle = model.forward_my_attack(X_input, c_trg, rand_param)
        adv_decoder_feats = model.get_decoder_features(X_input, c_trg, rand_param)

        model.zero_grad()
        return adv_output, adv_middle, adv_decoder_feats

    def get_atggan_features(self, X_input, c_trg, model, rand_param):
        # 获取干净特征
        with torch.no_grad():
            clean_output, _, _, clean_middle = model.forward_my_attack(X_input, c_trg, rand_param)
            clean_decoder_feats = model.get_decoder_features(X_input, c_trg, rand_param)

        # 获取对抗特征
        adv_output, _, _, adv_middle = model.forward_my_attack(X_input, c_trg, rand_param)
        adv_decoder_feats = model.get_decoder_features(X_input, c_trg, rand_param)

        # 增强解码器特征 - 针对注意力机制的特殊处理
        enhanced_decoder_features = []
        if isinstance(adv_decoder_feats, list) and isinstance(clean_decoder_feats, list):
            for i, (adv_feat, clean_feat) in enumerate(zip(adv_decoder_feats, clean_decoder_feats)):
                # 对不同层应用不同的增强策略
                if i < len(adv_decoder_feats) // 3:
                    # 前期层: 主要是卷积层输出，使用较大的扰动
                    diff = adv_feat - clean_feat
                    amplification = 3.0  # 增大扰动幅度
                    # 添加随机噪声增强攻击效果
                    noise = torch.randn_like(adv_feat) * 0.1
                    enhanced_feat = adv_feat + amplification * diff + noise
                    enhanced_decoder_features.append(enhanced_feat)
                elif i < 2 * len(adv_decoder_feats) // 3:
                    # 中期层: 注意力层激活，破坏注意力权重分布
                    diff = adv_feat - clean_feat
                    amplification = 2.5
                    # 对注意力特征应用非线性变换
                    attention_noise = torch.randn_like(adv_feat) * 0.15
                    enhanced_feat = adv_feat + amplification * diff + attention_noise
                    enhanced_decoder_features.append(enhanced_feat)
                else:
                    # 后期层: 保持相对稳定，避免过度扰动
                    diff = adv_feat - clean_feat
                    amplification = 1.8
                    enhanced_feat = adv_feat + amplification * diff
                    enhanced_decoder_features.append(enhanced_feat)
        else:
            enhanced_decoder_features = adv_decoder_feats

        model.zero_grad()
        return adv_output, adv_middle, enhanced_decoder_features

    def get_hisd_features(self, X_input, reference, F, T, G, E, gen, rand_param):
        # 获取干净特征
        with torch.no_grad():
            clean_c = E(X_input)  # Feature extractor
            clean_c_trg = clean_c
            s_trg = F(reference, 1)  # reference

            # attribute augmentation, default=No
            size = s_trg.size()
            q = rand_param
            s_trg = (s_trg * 0.8 + torch.tensor(np.random.uniform(s_trg.clone().detach().cpu().numpy().min(),
                                                                  s_trg.clone().detach().cpu().numpy().max(),
                                                                  size).astype('float32')).to(
                self.device) * 0.2) if q > 0.2 else s_trg

            clean_c_trg = T(clean_c_trg, s_trg, 1)
            clean_x_trg = G(clean_c_trg)
            clean_decoder_feats = gen.get_decoder_features(clean_c_trg, rand_param)

        # 获取对抗特征
        adv_c = E(X_input)  # Feature extractor
        adv_c_trg = adv_c
        adv_c_trg = T(adv_c_trg, s_trg, 1)
        adv_x_trg = G(adv_c_trg)
        adv_decoder_feats = gen.get_decoder_features(adv_c_trg, rand_param)

        gen.zero_grad()
        return adv_x_trg, adv_c_trg, adv_decoder_feats
