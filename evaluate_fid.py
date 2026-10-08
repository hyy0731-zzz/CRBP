import os

import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn.functional as F
from torch import nn
import numpy as np
from fid.src.pytorch_fid.fid_score import get_fid_score
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from attgan.data import check_attribute_conflict
from random import shuffle
import setGPU


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x


def compare(img1, img2):
    """input tensor, translate to np.array"""
    img1_np = img1.squeeze(0).cpu().numpy()
    img2_np = img2.squeeze(0).cpu().numpy()
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))

    ssim = structural_similarity(img1_np, img2_np, multichannel=True)
    psnr = peak_signal_noise_ratio(img1_np, img2_np)

    return ssim, psnr


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def evaluate_multiple_models(args_attack, test_dataloader, attgan, attgan_args, solver, attentiongan_solver, transform,
                             F, T, G, E, reference, gen_models, pgd_attack, max_samples, log_file='logs.txt', evaluate_fid=False):
    #  HiDF inference and evaluating
    l2_mask_error, fid_score, ssim_score, psnr_score = 0.0, 0.0, 0.0, 0.0
    SR_mask, n_samples = 0, 0
    save_img_num = 5
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        if idx == max_samples:
            break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a

        with torch.no_grad():
            # clean
            c = E(img_a)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen_noattack = G(c_trg)
            # adv

            current_noise = pgd_attack.up
            current_noise = torch.clamp(current_noise, -pgd_attack.epsilon, pgd_attack.epsilon)

            c = E(img_a + current_noise)
            c_trg = c
            s_trg = F(reference, 1)
            c_trg = T(c_trg, s_trg, 1)
            gen = G(c_trg)

            mask = abs(gen_noattack - img_a)
            mask = mask[0, 0, :, :] + mask[0, 1, :, :] + mask[0, 2, :, :]
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

            l2_mask_error += (((gen * mask - gen_noattack * mask) ** 2).sum() / (mask.sum() * 3))

            if evaluate_fid:
                os.makedirs('./gen', exist_ok=True)
                os.makedirs('./gen_noattack', exist_ok=True)
                vutils.save_image(gen, './gen/gen.jpg', nrow=10, normalize=True, range=(-1., 1.))
                vutils.save_image(gen_noattack, './gen_noattack/gen_noattack.jpg', nrow=10, normalize=True, range=(-1., 1.))
                fid_score += get_fid_score(['./gen', './gen_noattack'])
            if evaluate_fid and idx < save_img_num:
                os.makedirs('./gen/hisd', exist_ok=True)
                os.makedirs('./gen_noattack/hisd', exist_ok=True)
                vutils.save_image(gen, f'./gen/hisd/gen_{idx}.jpg', nrow=10, normalize=True, range=(-1., 1.))
                vutils.save_image(gen_noattack, f'./gen_noattack/hisd/gen_noattack_{idx}.jpg', nrow=10, normalize=True,
                                  range=(-1., 1.))

            ssim_local, psnr_local = compare(denorm(gen.clone()), denorm(gen_noattack.clone()))
            ssim_score += ssim_local
            psnr_score += psnr_local

            if (((gen * mask - gen_noattack * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                SR_mask += 1
            n_samples += 1

    print('HiDF {} images. L2_mask error: {}. SR_mask: {}. fid error: {}. psnr error: {}.  ssim error: {}.'.format(
        n_samples,
        l2_mask_error / n_samples,
        float(SR_mask) / n_samples,
        fid_score / n_samples,
        psnr_score / n_samples,
        ssim_score / n_samples))
    with open(log_file, 'a+') as f:
        f.write(
            'HiDF {} images. L2_mask error: {}. SR_mask: {}. fid error: {}. psnr error: {}.  ssim error: {}.'.format(
                n_samples,
                l2_mask_error / n_samples,
                float(SR_mask) / n_samples,
                fid_score / n_samples,
                psnr_score / n_samples,
                ssim_score / n_samples))
        f.write('\n')
    HiDF_prop_dist = float(SR_mask) / n_samples


    # AttGAN inference and evaluating
    l2_mask_error, fid_score, ssim_score, psnr_score = 0.0, 0.0, 0.0, 0.0
    SR_mask, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        if idx == max_samples:
            break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)
        att_b_list = [att_a]
        for i in range(attgan_args.n_attrs):
            tmp = att_a.clone()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, attgan_args.attrs[i], attgan_args.attrs)
            att_b_list.append(tmp)

        current_noise = pgd_attack.up
        current_noise = torch.clamp(current_noise, -pgd_attack.epsilon, pgd_attack.epsilon)

        samples = [img_a, img_a + current_noise]
        noattack_list = []

        shuffle(att_b_list)
        att_b_list = [att_b_list[0]]

        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * attgan_args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * attgan_args.test_int / attgan_args.thres_int
            with torch.no_grad():
                gen, _ = attgan.G(img_a + current_noise, att_b_)
                gen_noattack, _ = attgan.G(img_a, att_b_)

            samples.append(gen)
            noattack_list.append(gen_noattack)

            mask = abs(gen_noattack - img_a)
            mask = mask[0, 0, :, :] + mask[0, 1, :, :] + mask[0, 2, :, :]
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

            l2_mask_error += (((gen * mask - gen_noattack * mask) ** 2).sum() / (mask.sum() * 3))

            if evaluate_fid:
                os.makedirs('./gen', exist_ok=True)
                os.makedirs('./gen_noattack', exist_ok=True)
                vutils.save_image(gen, './gen/gen.jpg', nrow=10, normalize=True, range=(-1., 1.))
                vutils.save_image(gen_noattack, './gen_noattack/gen_noattack.jpg', nrow=10, normalize=True, range=(-1., 1.))
                fid_score += get_fid_score(['./gen', './gen_noattack'])
            if evaluate_fid and idx < save_img_num:
                os.makedirs('./gen/attgan', exist_ok=True)
                os.makedirs('./gen_noattack/attgan', exist_ok=True)
                vutils.save_image(gen, f'./gen/attgan/gen_{idx}.jpg', nrow=10, normalize=True, range=(-1., 1.))
                vutils.save_image(gen_noattack, f'./gen_noattack/attgan/gen_noattack_{idx}.jpg', nrow=10,
                                  normalize=True, range=(-1., 1.))

            ssim_local, psnr_local = compare(denorm(gen.clone()), denorm(gen_noattack.clone()))
            ssim_score += ssim_local
            psnr_score += psnr_local

            if (((gen * mask - gen_noattack * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                SR_mask += 1
            n_samples += 1
    print(
        'AttGAN {} images. L2_mask error: {}. SR_mask: {}. fid error: {}. psnr error: {}.  ssim error: {}.'.format(
        n_samples,
        l2_mask_error / n_samples,
        float(SR_mask) / n_samples,
        fid_score / n_samples,
        psnr_score / n_samples,
        ssim_score / n_samples))
    with open(log_file, 'a+') as f:
        f.write('AttGAN {} images. L2_mask error: {}. SR_mask: {}. fid error: {}. psnr error: {}.  ssim error: {}.'.format(
        n_samples,
        l2_mask_error / n_samples,
        float(SR_mask) / n_samples,
        fid_score / n_samples,
        psnr_score / n_samples,
        ssim_score / n_samples))
        f.write('\n')
    attgan_prop_dist = float(SR_mask) / n_samples


    # stargan inference and evaluating
    l2_mask_error, fid_score, ssim_score, psnr_score = 0.0, 0.0, 0.0, 0.0
    SR_mask, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        if idx == max_samples:
            break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)

        current_noise = pgd_attack.up
        current_noise = torch.clamp(current_noise, -pgd_attack.epsilon, pgd_attack.epsilon)

        x_noattack_list, x_fake_list = solver.test_universal_model_level(idx, img_a, c_org, current_noise,
                                                                         args_attack.stargan)

        number = len(x_noattack_list)
        rand_number = np.random.randint(0, number)
        x_noattack_list = [x_noattack_list[rand_number]]
        x_fake_list = [x_fake_list[rand_number]]

        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]

            mask = abs(gen_noattack - img_a)
            mask = mask[0, 0, :, :] + mask[0, 1, :, :] + mask[0, 2, :, :]
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

            l2_mask_error += (((gen * mask - gen_noattack * mask) ** 2).sum() / (mask.sum() * 3))

            if evaluate_fid:
                os.makedirs('./gen', exist_ok=True)
                os.makedirs('./gen_noattack', exist_ok=True)
                vutils.save_image(gen, './gen/gen.jpg', nrow=10, normalize=True, range=(-1., 1.))
                vutils.save_image(gen_noattack, './gen_noattack/gen_noattack.jpg', nrow=10, normalize=True, range=(-1., 1.))
                fid_score += get_fid_score(['./gen', './gen_noattack'])

            if evaluate_fid and idx < save_img_num:
                os.makedirs('./gen/stargan', exist_ok=True)
                os.makedirs('./gen_noattack/stargan', exist_ok=True)
                vutils.save_image(gen, f'./gen/stargan/gen_{idx}.jpg', nrow=10, normalize=True, range=(-1., 1.))
                vutils.save_image(gen_noattack, f'./gen_noattack/stargan/gen_noattack_{idx}.jpg', nrow=10,
                                  normalize=True, range=(-1., 1.))

            ssim_local, psnr_local = compare(denorm(gen.clone()), denorm(gen_noattack.clone()))
            ssim_score += ssim_local
            psnr_score += psnr_local

            if (((gen * mask - gen_noattack * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                SR_mask += 1
            n_samples += 1

    print('stargan {} images. L2_mask error: {}. SR_mask: {}. fid error: {}. psnr error: {}.  ssim error: {}.'.format(
        n_samples,
        l2_mask_error / n_samples,
        float(SR_mask) / n_samples,
        fid_score / n_samples,
        psnr_score / n_samples,
        ssim_score / n_samples))
    with open(log_file, 'a+') as f:
        f.write('stargan {} images. L2_mask error: {}. SR_mask: {}. fid error: {}. psnr error: {}.  ssim error: {}.'.format(
        n_samples,
        l2_mask_error / n_samples,
        float(SR_mask) / n_samples,
        fid_score / n_samples,
        psnr_score / n_samples,
        ssim_score / n_samples))
        f.write('\n')

    stargan_prop_dist = float(SR_mask) / n_samples

    # AttentionGAN inference and evaluating
    l2_mask_error, fid_score, ssim_score, psnr_score = 0.0, 0.0, 0.0, 0.0
    SR_mask, n_samples = 0, 0
    for idx, (img_a, att_a, c_org) in enumerate(test_dataloader):
        if idx == max_samples:
            break
        img_a = img_a.cuda() if args_attack.global_settings.gpu else img_a
        att_a = att_a.cuda() if args_attack.global_settings.gpu else att_a
        att_a = att_a.type(torch.float)

        current_noise = pgd_attack.up
        current_noise = torch.clamp(current_noise, -pgd_attack.epsilon, pgd_attack.epsilon)

        x_noattack_list, x_fake_list = attentiongan_solver.test_universal_model_level(idx, img_a, c_org, current_noise,
                                                                                      args_attack.AttentionGAN)
        number = len(x_noattack_list)
        rand_number = np.random.randint(0, number)
        x_noattack_list = [x_noattack_list[rand_number]]
        x_fake_list = [x_fake_list[rand_number]]

        for j in range(len(x_fake_list)):
            gen_noattack = x_noattack_list[j]
            gen = x_fake_list[j]

            mask = abs(gen_noattack - img_a)
            mask = mask[0, 0, :, :] + mask[0, 1, :, :] + mask[0, 2, :, :]
            mask[mask > 0.5] = 1
            mask[mask < 0.5] = 0

            l2_mask_error += (((gen * mask - gen_noattack * mask) ** 2).sum() / (mask.sum() * 3))

            if evaluate_fid:
                os.makedirs('./gen', exist_ok=True)
                os.makedirs('./gen_noattack', exist_ok=True)
                vutils.save_image(gen, './gen/gen.jpg', nrow=10, normalize=True, range=(-1., 1.))
                vutils.save_image(gen_noattack, './gen_noattack/gen_noattack.jpg', nrow=10, normalize=True, range=(-1., 1.))
                fid_score += get_fid_score(['./gen', './gen_noattack'])

            if evaluate_fid and idx < save_img_num:
                os.makedirs('./gen/attentiongan', exist_ok=True)
                os.makedirs('./gen_noattack/attentiongan', exist_ok=True)
                vutils.save_image(gen, f'./gen/attentiongan/gen_{idx}.jpg', nrow=10, normalize=True, range=(-1., 1.))
                vutils.save_image(gen_noattack, f'./gen_noattack/attentiongan/gen_noattack_{idx}.jpg', nrow=10,
                                  normalize=True, range=(-1., 1.))

            ssim_local, psnr_local = compare(denorm(gen.clone()), denorm(gen_noattack.clone()))
            ssim_score += ssim_local
            psnr_score += psnr_local

            if (((gen * mask - gen_noattack * mask) ** 2).sum() / (mask.sum() * 3)) > 0.05:
                SR_mask += 1
            n_samples += 1

    print('attentiongan {} images. L2_mask error: {}. SR_mask: {}. fid error: {}. psnr error: {}.  ssim error: {}.'.format(
        n_samples,
        l2_mask_error / n_samples,
        float(SR_mask) / n_samples,
        fid_score / n_samples,
        psnr_score / n_samples,
        ssim_score / n_samples))

    with open(log_file, 'a+') as f:
        f.write(
            'attentiongan {} images. L2_mask error: {}. SR_mask: {}. fid error: {}. psnr error: {}.  ssim error: {}.'.format(
        n_samples,
        l2_mask_error / n_samples,
        float(SR_mask) / n_samples,
        fid_score / n_samples,
        psnr_score / n_samples,
        ssim_score / n_samples))
        f.write('\n')

    aggan_prop_dist = float(SR_mask) / n_samples
    return HiDF_prop_dist, stargan_prop_dist, attgan_prop_dist, aggan_prop_dist
