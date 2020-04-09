import torch
from .dsr_model import DSRModel, base_SRModel
from .networks import AdaResBlock, AdaRCAGroup

class RCANModel(DSRModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            n_groups = 10,
            n_resblocks = 20,
            n_feats = 64,
            block_mode = 'CRC',
            channel_attention = 'ca',
        )
        return parser

    def __init__(self, opt):
        super(RCANModel, self).__init__(opt, SRModel=SRModel)
        self.model_names = ['RCAN']
        self.optimizer_names = ['RCAN_optimizer_%s' % opt.optimizer]
        self.netRCAN = self.netDSR


class SRModel(base_SRModel):
    def __init__(self, opt):
        self.block = AdaRCAGroup
        self.n_blocks = opt.n_groups
        self.block_name = 'group'
        super(SRModel, self).__init__(opt)