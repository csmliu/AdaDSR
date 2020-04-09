import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from . import losses as L

class Conv_ReLU_Block(nn.Module):
    def __init__(self, sparse_conv):
        super(Conv_ReLU_Block, self).__init__()
        self.sparse_conv = sparse_conv
        if sparse_conv:
            mode = 'c'
        else:
            mode = 'C'
        self.conv = N.conv(64, 64, 3, 1, 1, bias=False, mode=mode)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # mask = torch.ones((1, *x.shape[2:]), device=x.device)
        if self.sparse_conv:
            return self.relu(self.conv(x))#, mask))
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self, opt):
        super(VDSR, self).__init__()
        sparse_conv = opt.sparse_conv

        layers = []
        for ii in range(18):
            layers.append(Conv_ReLU_Block(sparse_conv))
        self.layers = layers
        self.residual_layer = N.seq(layers)

        self.input = N.conv(1, 64, 3, 1, 1, bias=False, mode='C')
        self.relu = nn.ReLU(True)
        self.output = N.conv(64, 1, 3, 1, 1, bias=False, mode='C')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.isTrain = opt.isTrain
        self.loss = opt.loss
        if self.isTrain:
            setattr(self, 'criterion%s'%self.loss,
                    getattr(L, '%sLoss'%self.loss)())

    def forward(self, x, hr=None, depth=None):
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out += x
        if self.isTrain:
            criterion1 = getattr(self, 'criterion%s'%self.loss)
            loss1 = criterion1(out, hr)
            return out, None, loss1, loss1
        return out, None


class VDSRModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            lr_mode = 'sr',
            mode = 'Y',
        )
        return parser

    def __init__(self, opt, SRModel=VDSR):
        super(VDSRModel, self).__init__(opt)

        self.opt = opt
        self.loss_names = [opt.loss, 'Total']
        self.visual_names = ['data_lr', 'data_hr', 'data_sr']
        self.model_names = ['DSR']
        self.optimizer_names = ['DSR_optimizer_%s' % opt.optimizer]

        DSR = SRModel(opt)
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
        self.data_lr = input['lr'].to(self.device) # save the Cx channels
        self.data_hr = input['hr'].to(self.device)
        self.data_lr_input = self.data_lr[:, :1, ...]
        self.data_hr_input = self.data_hr[:, :1, ...]
        self.image_paths = input['fname']

    def forward(self, FLOPs_only=False):
        if self.isTrain:
            self.data_sr_output, self.pred, *losses = \
                self.netDSR(self.data_lr_input, self.data_hr_input)
            for i, loss_name in enumerate(self.loss_names):
                setattr(self, 'loss_'+loss_name, losses[i].mean())
        else:
            if not self.opt.chop:
                self.data_sr_output, self.pred = self.netDSR(self.data_lr_input)
            else:
                self.data_sr_output, self.pred = N.forward_chop(
                        self.opt, self.netDSR, self.data_lr_input,
                        None, shave=10, min_size=160000)
        # Y channel is from the network output, while Cb and Cr channels are
        # from the tensor super resolved with bicubic algorithm.
        self.data_sr = self.data_lr.clone().detach()
        self.data_sr[:, :1, ...] = self.data_sr_output

    def backward(self):
        self.loss_Total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


