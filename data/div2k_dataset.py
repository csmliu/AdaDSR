import random
import numpy as np
import os
from os.path import join
from .sr_dataset import SRDataset

class DIV2KDataset(SRDataset):
    def __init__(self, opt, split='train', dataset_name='div2k'):
        super(DIV2KDataset, self).__init__(opt, split, dataset_name)
        if self.root == '':
            rootlist = ['D:/Datasets/SR/DIV2K',
                        '/data/DIV2K']
            for root in rootlist:
                if os.path.isdir(root):
                    self.root = root
                    break
        self.patch_size = opt.patch_size
        self.patch_size_lr = self.patch_size // self.scale
        assert self.patch_size % self.scale == 0
        self.hr_root = join(self.root, 'DIV2K_train_HR')
        self.lr_root = join(self.root, 'DIV2K_train_LR_bicubic/X%d'%self.scale)

        if split == 'train':
            self.start, self.num = 1, 800
            self._getitem = self._getitem_train
            self.len_data = self.num * (opt.test_every //
                                (self.num // self.batch_size))
        else:
            if split == 'val':
                self.start, self.num = 801, 5
            else:
                raise ValueError
            self._getitem = self._getitem_test
            self.len_data = self.num

        self.names = ['%04d'%i for i in range(self.start, self.start+self.num)]
        self.HR_images = [join(self.hr_root, '%s.png'%(n)) for n in self.names]
        self.LR_images = [join(self.lr_root, '%sx%d.png' % (n, self.scale)) \
                          for n in self.names]
        self.names = [join('DIV2K_%s'%split, i+'_SRBI_x%d.png'%self.scale) \
                      for i in self.names]

        self.load_data()

    def _crop(self, HR, LR):
        ih, iw = LR.shape[-2:]
        ix = random.randrange(0, iw - self.patch_size_lr + 1)
        iy = random.randrange(0, ih - self.patch_size_lr + 1)
        tx, ty = self.scale * ix, self.scale * iy
        return HR[..., ty:ty+self.patch_size, tx:tx+self.patch_size], \
               LR[..., iy:iy+self.patch_size_lr, ix:ix+self.patch_size_lr]

    def _augment_func(self, img, hflip, vflip, rot90):
        if hflip:   img = img[:, :, ::-1]
        if vflip:   img = img[:, ::-1, :]
        if rot90:   img = img.transpose(0, 2, 1) # CHW
        return np.ascontiguousarray(img)

    def _augment(self, *imgs):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        return (self._augment_func(img, hflip, vflip, rot90) for img in imgs)


if __name__ == '__main__':
    pass