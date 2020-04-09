import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Module
import functools
from torch.optim import lr_scheduler
from collections import OrderedDict

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            return 1 - max(0, epoch-opt.niter) / max(1, float(opt.niter_decay))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.niter,
                                                   eta_min=0)
    else:
        return NotImplementedError('lr [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 \
                or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, b=init_gain)
            else:
                raise NotImplementedError('[%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='default', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if init_type != 'default' and init_type is not None:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def forward_chop(opt, model, x, depth, shave=10, min_size=160000):
    scale = opt.scale
    pred = None
    n_GPUs = len(opt.gpu_ids)
    n_GPUs = 1 if n_GPUs == 0 else n_GPUs
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        pred_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch, pred_batch = model(lr_batch, depth=depth)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
            if pred_batch is not None:
                pred_list.extend(pred_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list, pred_list = [
            forward_chop(
                opt, model, patch, depth, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    if pred_batch is not None:
        pred = torch.empty(b, opt.nc_adapter, h, w)
        pred[:, :, 0:h_half, 0:w_half] = \
            pred_list[0][:, :, 0:h_half, 0:w_half]
        pred[:, :, 0:h_half, w_half:w] = \
            pred_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        pred[:, :, h_half:h, 0:w_half] = \
            pred_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        pred[:, :, h_half:h, w_half:w] = \
            pred_list[3][:, :, (h_size - h + h_half):h_size,
                               (w_size - w + w_half):w_size]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size,    
                           (w_size - w + w_half):w_size]

    return output, pred

class num_generator():
    def __init__(self, num_list):
        self.isTrain = True
        length = len(num_list)
        if length == 1:
            self.generate = self.generate_fix
            self.num = num_list[0]
        elif length == 2:
            self.generate = self.generate_rand
            self.min, self.max = num_list
        else:
            raise ValueError('num_list can only have 1 or 2 float numbers')

    def train(self, isTrain=True):
        self.isTrain = isTrain

    def eval(self):
        self.isTrain = False

    def generate_rand(self, *size, device=None):
        if not self.isTrain:
            return torch.empty(*size, device=device).fill_(self.max)
        return torch.empty(*size, device=device).uniform_(self.min, self.max)
        # return torch.rand(*size, device=device) * self.weight + self.bias
    def generate_fix(self, *size, device=None):
        return torch.empty(*size, device=device).fill_(self.num)

    def __call__(self, *size, device=None):
        return self.generate(*size, device=device)


'''
# ===================================
# Advanced nn.Sequential
# reform nn.Sequentials and nn.Modules
# to a single nn.Sequential
# ===================================
'''

def seq(*args):
    if len(args) == 1:
        args = args[0]
    if isinstance(args, nn.Module):
        return args
    modules = OrderedDict()
    if isinstance(args, OrderedDict):
        for k, v in args.items():
            modules[k] = seq(v)
        return nn.Sequential(modules)
    assert isinstance(args, (list, tuple))
    return nn.Sequential(*[seq(i) for i in args])

'''
# ===================================
# Useful blocks
# --------------------------------
# conv (+ normaliation + relu)
# concat
# sum
# resblock (ResBlock)
# resdenseblock (ResidualDenseBlock_5C)
# resinresdenseblock (RRDB)
# ===================================
'''


# -------------------------------------------------------
# Mask for `Partial Conv based pooling` (k3s1p1)
# -------------------------------------------------------
class mask(nn.Module):
    def __init__(self, feature_size):
        self.mask = torch.ones(feature_size, feature_size) * 1.5
        self.mask[1:-1, 1:-1] = 1
        self.mask[0, 0] = 2.25
        self.mask[0, -1] = 2.25
        self.mask[-1, 0] = 2.25
        self.mask[-1, -1] = 2.25
        self.mask.requires_grad = False

    def forward(self, x):
        return self.mask * x

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
         output_padding=0, dilation=1, groups=1, bias=True,
         padding_mode='zeros', mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'X':
            assert in_channels == out_channels
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'c':
            if 'MaskedConv2d' not in dir():
                from masked_conv2d import MaskedConv2d
                # Only Conv(k=3, s=1, p=1) is supported now
            L.append(MaskedConv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        output_padding=output_padding,
                                        groups=groups,
                                        bias=bias,
                                        dilation=dilation,
                                        padding_mode=padding_mode))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'i':
            L.append(nn.InstanceNorm2d(out_channels))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'P':
            L.append(nn.PReLU())
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return seq(*L)


class MeanShift(nn.Conv2d):
    """ is implemented via group conv """
    def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1, groups=3)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.ones(3).view(3, 1, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class MeanShift_rdn(nn.Conv2d):
    def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift_rdn, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


# -------------------------------------------------------
# Concat the output of a submodule to its input
# -------------------------------------------------------
class ConcatBlock(nn.Module):
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + '_concat'


# -------------------------------------------------------
# Elementwise sum the output of a submodule to its input
# -------------------------------------------------------
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# -------------------------------------------------------
# Content Unrelated Channel Attention (CUCA) Layer
# -------------------------------------------------------
class CUCALayer(nn.Module):
    def __init__(self, channel=64, min=0, max=None):
        super(CUCALayer, self).__init__()

        self.attention = nn.Conv2d(channel, channel, 1, padding=0,
                                   groups=channel, bias=False)
        self.min, self.max = min, max
        nn.init.uniform_(self.attention.weight, 0, 1)

    def forward(self, x):
        self.attention.weight.data.clamp_(self.min, self.max)
        return self.attention(x)


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding, bias, mode)

    def forward(self, x):
        res = self.res(x)
        return x + res


# -------------------------------------------------------
# Res Block suitable for multiple cases
# -------------------------------------------------------
class AdaResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, res_scale=1, bias=True, mode='CRC', reduction=16,
                 clamp=None, channel_attention='none', sparse_conv=False,
                 n_resblocks=None, clamp_wrapper=None, side_ca=False):
        super(AdaResBlock, self).__init__()
        # NOTE that `n_resblocks` and `clamp_wrapper` are unused

        assert in_channels == out_channels
        self.channel_attention = channel_attention
        assert channel_attention in ('none', '0', '1', 'ca')
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        # Predictor
        self.clamp = clamp
        self.side_ca = side_ca

        # Channel Attention (can be merged into the last conv)
        if channel_attention == 'none':
            self.cuca = lambda x: x #nn.Identity()
        elif channel_attention == '0':
            self.cuca = CUCALayer(out_channels, min=0, max=None)
        elif channel_attention == '1':
            self.cuca = CUCALayer(out_channels, min=0, max=1)
        elif channel_attention == 'ca':
            self.cuca = CALayer(out_channels, reduction)
        else:
            raise NotImplementedError

        # sparse_conv
        if sparse_conv:
            self.mode = mode.replace('C', 'c')
            self.d_weight = torch.ones(1, 1, 3, 3, device=0)
            self.d_weight.requires_grad_(False)
        else:
            self.mode = mode

        self.res_scale = res_scale
        for i, m in enumerate(self.mode):
            setattr(self, 'layer%d'%i, conv(
                in_channels, out_channels, kernel_size,
                stride, padding, bias=bias, mode=m))

    def forward(self, x, attention=None):
        if attention is None:
            d_map = torch.ones((1, 1, *x.shape[-2:]), device=x.device)
        else:
            d_map = self.clamp(attention)

        res = x.clone()
        dilation = True
        for i, m in enumerate(self.mode):
            if m == 'c':
                if dilation and attention is not None:
                    res = getattr(self, 'layer%d'%i)(res, F.conv2d(d_map,
                                  self.d_weight, padding=1)[0].detach())
                    dilation = False
                else:
                    res = getattr(self, 'layer%d'%i)(res, d_map[0])
            else:
                res = getattr(self, 'layer%d'%i)(res)
        if self.channel_attention == 'ca':
            res = self.cuca(res * d_map) * self.res_scale
        else:
            res = self.cuca(res) * self.res_scale * d_map
        return res + x


class AdaRCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, res_scale=1, bias=True, mode='CRC', reduction=16,
                 clamp=None, channel_attention='none', sparse_conv=False,
                 n_resblocks=20, clamp_wrapper=None, side_ca=False):
        super(AdaRCAGroup, self).__init__()
        # NOTE that `clamp` is unused
        assert in_channels == out_channels
        self.n_resblocks = n_resblocks
        if clamp_wrapper is None:
            clamp_wrapper = lambda x: None

        for i in range(self.n_resblocks):
            setattr(self, 'block%d'%i, AdaResBlock(
                    in_channels, out_channels, kernel_size, stride, padding,
                    res_scale, bias, mode, reduction, clamp_wrapper(i),
                    channel_attention, sparse_conv, side_ca=side_ca))
        setattr(self, 'block%d'%n_resblocks,
                conv(out_channels, out_channels, mode='C'))

    def forward(self, x, attention=None):
        res = x.clone()
        for i in range(self.n_resblocks):
            res = getattr(self, 'block%d'%i)(res, attention)
        res = getattr(self, 'block%d'%self.n_resblocks)(res)
        return res + x


# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding, bias=bias, mode=mode)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)# * res # NOTE: modified by multiplying res
        return res + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16, nb=12):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding,
                       bias, mode, reduction) for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))

        # self.rg = ShortcutBlock(nn.Sequential(*RG))
        self.rg = nn.Sequential(*RG)

    def forward(self, x):
        res = self.rg(x)
        return res + x


# -------------------------------------------------------
# Residual Dense Block
# style: 5 convs
# -------------------------------------------------------
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CR'):
        super(ResidualDenseBlock_5C, self).__init__()

        # gc: growth channel
        self.conv1 = conv(nc, gc, kernel_size, stride,
                          padding, bias=bias, mode=mode)
        self.conv2 = conv(nc+gc, gc, kernel_size, stride,
                          padding, bias=bias, mode=mode)
        self.conv3 = conv(nc+2*gc, gc, kernel_size, stride,
                          padding, bias=bias, mode=mode)
        self.conv4 = conv(nc+3*gc, gc, kernel_size, stride,
                          padding, bias=bias, mode=mode)
        self.conv5 = conv(nc+4*gc, nc, kernel_size, stride,
                          padding, bias=bias, mode=mode[:-1])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


# -------------------------------------------------------
# Residual in Residual Dense Block
# 3x5c
# -------------------------------------------------------
class RRDB(nn.Module):
    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CR'):
        super(RRDB, self).__init__()

        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride,
                                          padding, bias, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride,
                                          padding, bias, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride,
                                          padding, bias, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


'''
# ======================
# Upsampler
# ======================
'''


# -------------------------------------------------------
# conv + subp + relu
# -------------------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3,
                          stride=1, padding=1, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 4BR.
    assert len(mode)<4 and mode[0] in ['2', '3', '4']
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size,
               stride, padding, bias=bias, mode='C'+mode)
    return up1


# -------------------------------------------------------
# nearest_upsample + conv + relu
# -------------------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1,
                    padding=1, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 3BR.
    assert len(mode)<4 and mode[0] in ['2', '3']
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride,
               padding, bias=bias, mode=mode)
    return up1


# -------------------------------------------------------
# convTranspose + relu
# -------------------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2,
                           stride=2, padding=0, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 4BR.
    assert len(mode)<4 and mode[0] in ['2', '3', '4']
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride,
               padding, bias=bias, mode=mode)
    return up1


'''
# ======================
# Downsampler
# ======================
'''


# -------------------------------------------------------
# strideconv + relu
# -------------------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2,
                          stride=2, padding=0, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 4BR.
    assert len(mode)<4 and mode[0] in ['2', '3', '4']
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride,
                 padding, bias=bias, mode=mode)
    return down1


# -------------------------------------------------------
# maxpooling + conv + relu
# -------------------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3,
                       stride=1, padding=0, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 3BR.
    assert len(mode)<4 and mode[0] in ['2', '3']
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride,
                     padding, bias=bias, mode=mode[1:])
    return sequential(pool, pool_tail)


# -------------------------------------------------------
# averagepooling + conv + relu
# -------------------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3,
                       stride=1, padding=1, bias=True, mode='2R'):
    # mode examples: 2, 2R, 2BR, 3, ..., 3BR.
    assert len(mode)<4 and mode[0] in ['2', '3']
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride,
                     padding, bias=bias, mode=mode[1:])
    return sequential(pool, pool_tail)

