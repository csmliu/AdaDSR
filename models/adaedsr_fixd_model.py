import torch
from .dsr_model import DSRModel, base_SRModel
from .networks import AdaResBlock, AdaRCAGroup

class AdaEDSRFixDModel(DSRModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            n_resblocks = 32,
            n_feats = 256,
            block_mode = 'CRC',
            nc_adapter = 1,
            constrain = 'soft',
            depth = [1],
            adapter_reduction = 2,
        )
        return parser

    def __init__(self, opt):
        super(AdaEDSRFixDModel, self).__init__(opt, SRModel=SRModel)
        assert len(opt.depth) == 1
        self.model_names = ['AdaEDSRFixD']
        self.optimizer_names = ['AdaEDSRFixD_optimizer_%s' % opt.optimizer]
        self.netAdaEDSRFixD = self.netDSR


class SRModel(base_SRModel):
    def __init__(self, opt):
        self.block = AdaResBlock
        self.n_blocks = opt.n_resblocks
        self.block_name = 'block'
        super(SRModel, self).__init__(opt)