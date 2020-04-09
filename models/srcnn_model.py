import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from . import losses as L

class SRCNN(nn.Module):
    def __init__(self, opt):
        super(SRCNN, self).__init__()
        self.conv1 = N.conv(1, 64, 9, padding=4, mode='CR')
        self.conv2 = N.conv(64, 32, 5, padding=2, mode='CR')
        self.conv3 = N.conv(32, 1, 5, padding=2, mode='C')

        self.isTrain = opt.isTrain
        self.loss = opt.loss
        if self.isTrain:
            setattr(self, 'criterion%s'%self.loss,
                    getattr(L, '%sLoss'%self.loss)())

    def forward(self, x, hr=None, depth=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.isTrain:
            criterion1 = getattr(self, 'criterion%s'%self.loss)
            loss1 = criterion1(x, hr)
            return x, None, loss1, loss1
        return x, None

class SRCNNModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            lr_mode = 'sr',
            mode = 'Y',
        )
        return parser

    def __init__(self, opt, SRModel=SRCNN):
        super(SRCNNModel, self).__init__(opt)

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

    def forward(self):
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

