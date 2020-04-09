import torch
from .dsr_model import DSRModel, base_SRModel
from .networks import AdaResBlock, AdaRCAGroup

class AdaEDSRModel(DSRModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            n_resblocks = 32,
            n_feats = 256,
            block_mode = 'CRC',
            nc_adapter = 1,
            constrain = 'soft',
            depth = [0, 32],
            adapter_pos = 5,
            adapter_reduction = 2,
        )
        return parser

    def __init__(self, opt):
        super(AdaEDSRModel, self).__init__(opt, SRModel=SRModel)
        assert self.isTrain and len(opt.depth) == 2 or \
           not self.isTrain and len(opt.depth) == 1
        self.model_names = ['AdaEDSR']
        self.optimizer_names = ['AdaEDSR_optimizer_%s' % opt.optimizer]
        self.netAdaEDSR = self.netDSR


class SRModel(base_SRModel):
    def __init__(self, opt):
        self.block = AdaResBlock
        self.n_blocks = opt.n_resblocks
        self.block_name = 'block'
        super(SRModel, self).__init__(opt)