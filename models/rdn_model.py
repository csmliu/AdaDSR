import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from . import losses as L

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, sparse_conv):
        super(RDB_Conv, self).__init__()
        mode = 'CR'
        if sparse_conv:
            mode = mode.replace('C', 'c')
        self.conv = N.conv(inChannels, growRate, 3, 1, 1, mode=mode)

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, sparse_conv):
        super(RDB, self).__init__()
        G0 = growRate0 # 64
        G = growRate # 64
        C = nConvLayers # 8

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G, sparse_conv))
        self.convs = N.seq(convs)

        # Local Feature Fusion
        self.LFF = N.conv(G0 + C * G, G0, 1, stride=1, padding=0, mode='C')

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, opt):
        super(RDN, self).__init__()
        input_nc = 3
        r = opt.scale
        sparse_conv = opt.sparse_conv
        G0 = 64

        # number of RDB blocks, conv layers, out channels
        RDNconfig  ='B'
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[RDNconfig]

        self.sub_mean = N.MeanShift_rdn()
        # Shallow feature extraction net
        self.SFENet1 = N.conv(input_nc, G0, 3, 1, 1, mode='C')
        self.SFENet2 = N.conv(G0, G0, 3, 1, 1, mode='C')

        # Redidual dense blocks and dense feature fusion
        for i in range(self.D):
            setattr(self, 'RDB%d'%i, RDB(G0, G, C, sparse_conv))

        # Global Feature Fusion
        self.GFF = N.seq(
            N.conv(self.D * G0, G0, 1, stride=1, padding=0, mode='C'),
            N.conv(G0, G0, 3, 1, 1, mode='C')
        )

        # Up-sampling net
        UPNet = []
        if r == 2 or r == 3:
            UPNet.append(N.upsample_pixelshuffle(G0, G, 3, 1, 1, mode=str(r)))
        elif r == 4:
            UPNet.append(N.upsample_pixelshuffle(G0, G, 3, 1, 1, mode='2'))
            UPNet.append(N.upsample_pixelshuffle(G, G, 3, 1, 1, mode='2'))
        else:
            raise ValueError("scale must be 2 or 3 or 4.")
        UPNet.append(N.conv(G, input_nc, 3, 1, 1, mode='C'))
        self.UPNet = N.seq(UPNet)

        self.add_mean = N.MeanShift_rdn(sign=1)

        self.isTrain = opt.isTrain
        self.loss = opt.loss
        if self.isTrain:
            setattr(self, 'criterion%s'%self.loss,
                    getattr(L, '%sLoss'%self.loss)())

    def forward(self, x, hr=None, depth=None):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = getattr(self, 'RDB%d'%i)(x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        x = self.UPNet(x)
        x = self.add_mean(x)
        if self.isTrain:
            criterion1 = getattr(self, 'criterion%s'%self.loss)
            loss1 = criterion1(x, hr)
            return x, None, loss1, loss1
        return x, None




class RdnModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt, SRModel=RDN):
        super(RdnModel, self).__init__(opt)

        self.opt = opt
        self.loss_names = [opt.loss, 'Total']
        self.visual_names = ['data_lr', 'data_hr', 'data_sr']
        self.model_names = ['DSR']
        self.optimizer_names = ['DSR_optimizer_%s' % opt.optimizer]

        DSR = RDN(opt)
        self.netDSR = N.init_net(DSR, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:
            if opt.optimizer == 'Adam':
                self.optimizer = optim.Adam(self.netDSR.parameters(),
                                            lr=opt.lr,
                                            betas=(opt.beta1, opt.beta2),
                                            weight_decay=opt.weight_decay)
            elif opt.optimizer == 'SGD':
                self.optimizer = optim.SGD(self.netDSR.parameters(),
                                           lr=opt.lr,
                                           momentum=opt.momentum,
                                           weight_decay=opt.weight_decay)
            elif opt.optimizer == 'RMSprop':
                self.optimizer = optim.RMSprop(self.netDSR.parameters(),
                                               lr=opt.lr,
                                               alpha=opt.alpha,
                                               momentum=opt.momentum,
                                               weight_decay=opt.weight_decay)
            else:
                raise NotImplementedError(
                    'optimizer named [%s] is not supported' % opt.optimizer)

            self.optimizers = [self.optimizer]

    def set_input(self, input):
        self.data_lr = input['lr'].to(self.device)
        self.data_hr = input['hr'].to(self.device)
        self.image_paths = input['fname']

    def forward(self, FLOPs_only=False):
        if self.isTrain:
            self.data_sr, self.pred, *losses = \
                self.netDSR(self.data_lr, self.data_hr)
            for i, loss_name in enumerate(self.loss_names):
                setattr(self, 'loss_'+loss_name, losses[i].mean())
        else:
            if not self.opt.chop:
                self.data_sr, self.pred = self.netDSR(self.data_lr)
            else:
                self.data_sr, self.pred = N.forward_chop(
                        self.opt, self.netDSR, self.data_lr,
                        None, shave=10, min_size=160000)

    def backward(self):
        self.loss_L1.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()