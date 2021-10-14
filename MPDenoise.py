import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict
import cv2


def conv(in_c, out_c, kernel_size):
    return nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=1, stride=1)

class DownSample(nn.Module):
    def __init__(self, in_c, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c, in_c + s_factor, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_c, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_c+s_factor, in_c, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.CA = CALayer(n_feat, reduction)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, 3, kernel_size)
        self.conv3 = conv(3, n_feat, kernel_size)
    def forward(self, x, x_img):
        # x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        # x2 = torch.sigmoid(self.conv3(img))
        # x1 = x1 * x2
        # x1 = x1 + x
        return img


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, scale_unetfeats, act):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, act) for _ in range(2)]
        self.encoder_level4 = [CAB(n_feat + scale_unetfeats * 3, kernel_size, reduction, act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.encoder_level4 = nn.Sequential(*self.encoder_level4)

        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)
        self.down34 = DownSample(n_feat + scale_unetfeats*2, scale_unetfeats)

    def forward(self, x):
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        x = self.down34(enc3)
        enc4 = self.encoder_level4(x)
        return [enc1, enc2, enc3, enc4]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, scale_unetfeats, act):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat + scale_unetfeats*2, kernel_size, reduction, act) for _ in range(2)]
        self.decoder_level4 = [CAB(n_feat + scale_unetfeats*3, kernel_size, reduction, act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        self.decoder_level4 = nn.Sequential(*self.decoder_level4)

        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, act)
        self.skip_attn3 = CAB(n_feat + scale_unetfeats*2, kernel_size, reduction, act)

        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)
        self.up43 = SkipUpSample(n_feat + scale_unetfeats*2, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3, enc4 = outs
        dec4 = self.decoder_level4(enc4)

        x = self.up43(dec4, self.skip_attn3(enc3))
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3, dec4]


# class Encoder(nn.Module):
#     def __init__(self, n_feat, kernel_size, reduction, scale_unetfeats, act):
#         super(Encoder, self).__init__()
#
#         self.encoder_level1 = [CAB(n_feat, kernel_size, reduction, act) for _ in range(2)]
#         self.encoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, act) for _ in range(2)]
#         self.encoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, act) for _ in range(2)]
#
#         self.encoder_level1 = nn.Sequential(*self.encoder_level1)
#         self.encoder_level2 = nn.Sequential(*self.encoder_level2)
#         self.encoder_level3 = nn.Sequential(*self.encoder_level3)
#
#         self.down12 = DownSample(n_feat, scale_unetfeats)
#         self.down23 = DownSample(n_feat+scale_unetfeats, scale_unetfeats)
#
#     def forward(self, x):
#         enc1 = self.encoder_level1(x)
#         x = self.down12(enc1)
#         enc2 = self.encoder_level2(x)
#         x = self.down23(enc2)
#         enc3 = self.encoder_level3(x)
#         return [enc1, enc2, enc3]
#
# class Decoder(nn.Module):
#     def __init__(self, n_feat, kernel_size, reduction, scale_unetfeats, act):
#         super(Decoder, self).__init__()
#
#         self.decoder_level1 = [CAB(n_feat, kernel_size, reduction, act) for _ in range(2)]
#         self.decoder_level2 = [CAB(n_feat + scale_unetfeats, kernel_size, reduction, act) for _ in range(2)]
#         self.decoder_level3 = [CAB(n_feat + scale_unetfeats * 2, kernel_size, reduction, act) for _ in range(2)]
#
#         self.decoder_level1 = nn.Sequential(*self.decoder_level1)
#         self.decoder_level2 = nn.Sequential(*self.decoder_level2)
#         self.decoder_level3 = nn.Sequential(*self.decoder_level3)
#
#         self.skip_attn1 = CAB(n_feat, kernel_size, reduction, act)
#         self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, act)
#
#         self.up21 = SkipUpSample(n_feat, scale_unetfeats)
#         self.up32 = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)
#
#     def forward(self, outs):
#         enc1, enc2, enc3 = outs
#         dec3 = self.decoder_level3(enc3)
#
#         x = self.up32(dec3, self.skip_attn2(enc2))
#         dec2 = self.decoder_level2(x)
#
#         x = self.up21(dec2, self.skip_attn1(enc1))
#         dec1 = self.decoder_level1(x)
#
#         return [dec1, dec2, dec3]


class WlrDenoiseModel(nn.Module):
    def __init__(self, in_c=3, n_feat=80, kernel_size=3, reduction=4, scale_unetfeats=48):
        super(WlrDenoiseModel, self).__init__()
        act = nn.ReLU()
        self.shallow_feat = nn.Sequential(conv(in_c, n_feat, kernel_size), CAB(n_feat, kernel_size, reduction, act))

        self.encoder = Encoder(n_feat, kernel_size, reduction, scale_unetfeats, act)
        self.decoder = Decoder(n_feat, kernel_size, reduction, scale_unetfeats, act)

        self.conv = conv(n_feat, in_c, kernel_size)

    def forward(self, x_img):
        x1 = self.shallow_feat(x_img)
        x_feat = self.encoder(x1)
        res = self.decoder(x_feat)

        restored_x = x_img + self.conv(res[0])
        return restored_x

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([0.6]), torch.tensor([0.6]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs, 1)).view(-1, 1, 1, 1).cuda()

        rgb_gt = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20 * np.log10(255/rmse)
    return ps

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
