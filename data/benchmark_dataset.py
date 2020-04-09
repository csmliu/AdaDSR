import os
import cv2
import numpy as np
from os.path import join
from .sr_dataset import SRDataset

class BenchmarkDataset(SRDataset):

    name2dir = {'div2k': 'DIV2K_valid_HR', 'set5': 'Set5', 'set14': 'Set14',
                'b100': 'B100', 'urban100': 'Urban100', 'manga109': 'Manga109'}

    def __init__(self, opt, split, dataset_name):
        super(BenchmarkDataset, self).__init__(opt, split, dataset_name)
        if self.root == '':
            rootlist = ['D:/Datasets/SR/SR',
                        '/data/SR']
            for root in rootlist:
                if os.path.isdir(root):
                    self.root = root
                    break
        self.hr_root = join(self.root, 'HR/%s/x%d' % \
                            (self.name2dir[self.dataset_name], self.scale))
        self.lr_root = join(self.root, 'LR/LRBI/%s/x%d' % \
                            (self.name2dir[self.dataset_name], self.scale))

        if split == 'test':
            self.HR_images, self.LR_images, self.names = self._scan()
            self._getitem = self._getitem_test
            self.num = self.len_data = len(self.names)
        else:
            raise ValueError
        self.load_data()

    def _scan(self):
        fnames = []
        list_hr = []
        list_lr = []
        for filename in os.listdir(self.hr_root):
            if not self.imio.is_image(filename):  continue
            list_hr.append(join(self.hr_root, filename))
            *fname, _, ext = filename.split('_')    # e.g., 0801_HR_x2.png
            fname = '_'.join(fname)
            fnames.append(join(self.dataset_name, fname + '_SRBI_' + ext))
            list_lr.append(join(self.lr_root, fname + '_LRBI_' + ext))
        return list_hr, list_lr, fnames


if __name__ == '__main__':
    pass
