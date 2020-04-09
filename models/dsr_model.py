import torch
from .base_model import BaseModel
from . import networks as N
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from . import losses as L


class DSRModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt, SRModel=None):
        super(DSRModel, self).__init__(opt)

        self.opt = opt
        self.nc_adapter = opt.nc_adapter
        self.constrain = opt.constrain

        if self.nc_adapter != 0:
            self.loss_names = [opt.loss, 'Pred', 'Total']
            self.visual_names = ['data_lr', 'data_hr', 'data_sr', 'pred']
        else:
            self.loss_names = [opt.loss, 'Total']
            self.visual_names = ['data_lr', 'data_hr', 'data_sr']
        self.model_names = ['DSR'] # will rename in subclasses
        self.optimizer_names = ['DSR_optimizer_%s' % opt.optimizer]

        DSR = SRModel(opt)
        self.netDSR = N.init_net(DSR, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.constrain != 'none':
            self.depth_gen = N.num_generator(opt.depth)
        else:
            self.depth_gen = None
            self.depth = None

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
        if self.depth_gen is not None:
            batch_size = self.data_lr.shape[0]
            self.depth = self.depth_gen((batch_size, 1), device=self.device)

    def forward(self, FLOPs_only=False):
        if self.isTrain:
            self.data_sr, self.pred, *losses = \
                self.netDSR(self.data_lr, self.data_hr, self.depth, FLOPs_only)
            for i, loss_name in enumerate(self.loss_names):
                setattr(self, 'loss_'+loss_name, losses[i].mean())
        elif self.opt.model.lower() in ('adaedsr', 'adarcan', 'adaedsr_fixd'):
            # We write a chop function for AdaEDSR and AdaRCAN for running
            # the adapter only once, see `class base_SRModel` for details.
            self.data_sr, self.pred = self.netDSR(self.data_lr,
                                                  depth=self.depth,
                                                  FLOPs_only=FLOPs_only,
                                                  chop=self.opt.chop)
        else:
            if not self.opt.chop:
                self.data_sr, self.pred = self.netDSR(self.data_lr,
                                                      depth=self.depth)
            else:
                self.data_sr, self.pred = N.forward_chop(self.opt, self.netDSR,
                        self.data_lr, self.depth, shave=10, min_size=160000)

    def backward(self):
        self.loss_Total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


class base_SRModel(nn.Module):
    def __init__(self, opt):
        super(base_SRModel, self).__init__()

        self.opt = opt
        self.lambda_pred = opt.lambda_pred
        self.nc_adapter = opt.nc_adapter
        self.multi_adapter = opt.multi_adapter
        self.constrain = opt.constrain
        self.with_depth = opt.with_depth
        self.scale = opt.scale

        if self.nc_adapter > 0 and self.multi_adapter:
            assert self.n_blocks == self.nc_adapter

        n_feats = opt.n_feats
        n_upscale = int(math.log(opt.scale, 2))

        m_head = [N.MeanShift(),
                  N.conv(opt.input_nc, n_feats, mode='C')]
        self.head = N.seq(m_head)

        for i in range(self.n_blocks):
            setattr(self, '%s%d'%(self.block_name, i), self.block(
                n_feats, n_feats, res_scale=opt.res_scale, mode=opt.block_mode,
                clamp=self.clamp_wrapper(i) if self.nc_adapter != 0 else None,
                channel_attention=opt.channel_attention,
                sparse_conv=opt.sparse_conv,
                n_resblocks=opt.n_resblocks,
                clamp_wrapper=self.clamp_wrapper,
                side_ca=opt.side_ca))
            if self.nc_adapter != 0 and self.multi_adapter:
                setattr(self, 'predictor%d'%i, Predictor(
                    n_feats=n_feats, n_layers=opt.adapter_layers,
                    reduction=opt.adapter_reduction,
                    hard_constrain=(self.constrain=='hard'),
                    nc_adapter=1,
                    depth_pos=opt.adapter_pos,
                    upper_bound=opt.adapter_bound))
        self.body_lastconv = N.conv(n_feats, n_feats, mode='C')

        if opt.scale == 3:
            m_up = N.upsample_pixelshuffle(n_feats, n_feats, mode='3')
        else:
            m_up = [N.upsample_pixelshuffle(n_feats, n_feats, mode='2') \
                    for _ in range(n_upscale)]
        self.up = N.seq(m_up)

        m_tail = [N.conv(n_feats, opt.output_nc, mode='C'),
                  N.MeanShift(sign=1)]
        self.tail = N.seq(m_tail)

        if self.nc_adapter != 0 and not self.multi_adapter:
            assert self.nc_adapter in (1, self.n_blocks)
            self.predictor = Predictor(
                    n_feats=n_feats, n_layers=opt.adapter_layers,
                    reduction=opt.adapter_reduction,
                    hard_constrain=(self.constrain=='hard'),
                    nc_adapter=self.nc_adapter,
                    depth_pos=opt.adapter_pos,
                    upper_bound=opt.adapter_bound)

        self.isTrain = opt.isTrain
        self.loss = opt.loss
        if self.isTrain:
            setattr(self, 'criterion%s'%self.loss,
                    getattr(L, '%sLoss'%self.loss)())

    def clamp_wrapper(self, i):
        def clamp(x):
            return torch.clamp(x-i, 0, 1)
        return clamp

    def forward_main_tail(self, x, pred):
        res = x
        for i in range(self.n_blocks):
            if self.nc_adapter <= 1 and not self.multi_adapter:
                res = getattr(self, '%s%d'%(self.block_name, i))(
                        res, pred)
            elif self.multi_adapter:
                setattr(self, 'pred%d'%i,
                        getattr(self, 'predictor%d'%i)(res,
                                depth if self.with_depth else None))
                res = getattr(self, '%s%d'%(self.block_name, i))(
                        res, getattr(self, 'pred%d'%i))
            else:
                res = getattr(self, '%s%d'%(self.block_name, i))(
                        res, pred[:, i:i+1, ...])
        res = self.body_lastconv(res)
        res += x

        res = self.up(res)
        res = self.tail(res)
        return res

    def forward_chop(self, x, pred, shave=10, min_size=160000):
        scale = self.scale
        n_GPUs = len(self.opt.gpu_ids)
        n, c, h, w = x.shape
        h_half, w_half = h//2, w//2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]
        pred_list = [
            pred[..., 0:h_size, 0:w_size],
            pred[..., 0:h_size, (w - w_size):w],
            pred[..., (h - h_size):h, 0:w_size],
            pred[..., (h - h_size):h, (w - w_size):w]
        ]
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i+n_GPUs)], dim=0)
                pred_batch = torch.cat(pred_list[i:(i+n_GPUs)], dim=0)
                res = self.forward_main_tail(lr_batch, pred_batch)
                sr_list.extend(res.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(lr_, pred_, shave, min_size) \
                    for lr_, pred_ in zip(lr_list, pred_list)]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
        c = sr_list[0].shape[1]

        output = x.new(n, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size,
                               (w_size - w + w_half):w_size]
        return output



    def forward(self, x, hr=None, depth=None, FLOPs_only=False, chop=False):
        x = self.head(x)

        if not self.multi_adapter:
            if self.nc_adapter:
                if self.with_depth:
                    pred = self.predictor(x, depth) # N*1*H*W, and depth is N*1
                else:
                    pred = self.predictor(x)
            else:
                pred = None
        if FLOPs_only:
            return x, pred

        if chop:
            x = self.forward_chop(x, pred)
        else:
            x = self.forward_main_tail(x, pred)

        if self.isTrain:
            criterion1 = getattr(self, 'criterion%s'%self.loss)
            loss1 = criterion1(x, hr)
            if self.nc_adapter != 0:
                if self.constrain == 'none':
                    loss_Pred = self.lambda_pred * pred.abs()
                    loss = loss1 + loss_Pred
                elif self.constrain == 'soft':
                    if self.multi_adapter:
                        pred = torch.cat([getattr(self, 'pred%d'%i) \
                             for i in range(self.nc_adapter)], dim=1)
                        loss_Pred = self.lambda_pred * \
                            (pred.mean((2,3)) - depth).clamp_min_(0).sum(dim=1)
                    else:
                        loss_Pred = self.lambda_pred * \
                            (pred.mean((2,3)) - depth).clamp_min_(0).mean(dim=1)
                            #(pred.mean((1,2,3)) - depth).clamp_min_(0)
                    # loss_Pred = self.lambda_pred * \
                    #             (pred.mean((1,2,3)) - depth).abs()
                    loss = loss1 + loss_Pred
                else:
                    loss = loss1
                    loss_Pred = torch.zeros_like(loss1)
                return x, pred, loss1, loss_Pred, loss
            return x, pred, loss1, loss1
        else:
            if self.multi_adapter:
                pred = torch.cat([getattr(self, 'pred%d'%i) \
                                  for i in range(self.nc_adapter)], dim=1)
        return x, pred

class Predictor(nn.Module):
    def __init__(self, n_feats, n_layers=5, reduction=2, hard_constrain=False,
                 nc_adapter=1, depth_pos=-1, upper_bound=float('inf')):
        super(Predictor, self).__init__()

        self.hard_constrain = hard_constrain
        self.depth_pos = depth_pos
        self.upper_bound = upper_bound
        self.n_layers = n_layers

        pred_feats = n_feats // reduction
        layers = [
            N.conv(n_feats, pred_feats, 3, mode='C'),
            *(N.conv(pred_feats, pred_feats, 3, mode='PC') \
                    for _ in range(n_layers - 2)),
            N.conv(pred_feats, nc_adapter, 3, mode='PC')
        ]
        for i, layer in enumerate(layers):
            setattr(self, 'layer%d'%i, layer)

    def forward(self, x, depth=None):
        for i in range(self.n_layers):
            if self.depth_pos == i:
                x = x * depth.view(-1, 1, 1, 1)
            x = getattr(self, 'layer%d'%i)(x)
        if self.depth_pos >= self.n_layers:
            x = x * depth.view(-1, 1, 1, 1)
        if self.hard_constrain:
            return x / x.mean((1, 2, 3), keepdim=True) * depth.view(-1, 1, 1, 1)
        return x.clamp(0, self.upper_bound)
