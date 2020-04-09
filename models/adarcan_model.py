import torch
from .dsr_model import DSRModel, base_SRModel
from .networks import AdaResBlock, AdaRCAGroup

class AdaRCANModel(DSRModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            n_groups = 10,
            n_resblocks = 20,
            n_feats = 64,
            block_mode = 'CRC',
            channel_attention = 'ca',
            nc_adapter = 10,
            constrain = 'soft',
            adapter_pos = 0,
            adapter_reduction = 1,
            depth = [0.1, 20],
            lambda_pred = 0.03,
        )
        return parser

    def __init__(self, opt):
        super(AdaRCANModel, self).__init__(opt, SRModel=SRModel)
        assert self.isTrain and len(opt.depth) == 2 or \
           not self.isTrain and len(opt.depth) == 1
        self.model_names = ['AdaRCAN']
        self.optimizer_names = ['AdaRCAN_optimizer_%s' % opt.optimizer]
        self.netAdaRCAN = self.netDSR


class SRModel(base_SRModel):
    def __init__(self, opt):
        self.block = AdaRCAGroup
        self.n_blocks = opt.n_groups
        self.block_name = 'group'
        super(SRModel, self).__init__(opt)
