import torch
from .dsr_model import DSRModel, base_SRModel
from .networks import AdaResBlock, AdaRCAGroup

class SRResNetModel(DSRModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            n_resblocks = 16,
            n_feats = 64,
            block_mode = 'CBRCB'
        )
        return parser

    def __init__(self, opt):
        super(SRResNetModel, self).__init__(opt, SRModel=SRModel)
        self.model_names = ['SRResNet']
        self.optimizer_names = ['SRResNet_optimizer_%s' % opt.optimizer]
        self.netSRResNet = self.netDSR


class SRModel(base_SRModel):
    def __init__(self, opt):
        self.block = AdaResBlock
        self.n_blocks = opt.n_resblocks
        self.block_name = 'block'
        super(SRModel, self).__init__(opt)