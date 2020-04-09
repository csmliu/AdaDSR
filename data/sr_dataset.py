import os
import cv2
import random
import numpy as np
from .imlib import imlib
from os.path import join
from data.base_dataset import BaseDataset

class SRDataset(BaseDataset):
    def __init__(self, opt, split, dataset_name):
        super(SRDataset, self).__init__(opt, split, dataset_name)
        self.mode = opt.mode  # RGB, Y or L
        self.imio = imlib(self.mode, lib=opt.imlib)
        self.scale = opt.scale
        self.preload = opt.preload
        self.batch_size = opt.batch_size
        self.lr_mode = opt.lr_mode
        if self.lr_mode == 'lr':
            self.lr_process = lambda lr_img: lr_img.astype(np.float32)
        else:
            self.lr_process = self.lr_process_sr

        self.getimage = self.getimage_read
        self.multi_imreader = opt.multi_imreader

    def load_data(self):
        if self.preload:
            if self.multi_imreader:
                read_images(self)
            else:
                self.HR_images = [self.imio.read(p) for p in self.HR_images]
                self.LR_images = [self.imio.read(p) for p in self.LR_images]
            self.getimage = self.getimage_preload

    def lr_process_sr(self, lr_img):
        if lr_img.shape[0] == 1:
            return np.expand_dims(cv2.resize(lr_img[0].astype(np.float32),
                                  dsize=(0, 0), fx=self.scale, fy=self.scale,
                                  interpolation=cv2.INTER_CUBIC), 0)
        return cv2.resize(lr_img.transpose(1, 2, 0).astype(np.float32),
                          dsize=(0, 0), fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

    def getimage_preload(self, index):
        return self.HR_images[index], self.LR_images[index], self.names[index]

    def getimage_read(self, index):
        return self.imio.read(self.HR_images[index]), \
               self.imio.read(self.LR_images[index]), self.names[index]


    def _getitem_train(self, index):
        index = index % self.num
        hr_img, lr_img, f_name = self.getimage(index)
        hr_img, lr_img = self._crop(hr_img, lr_img)
        hr_img, lr_img = self._augment(hr_img, lr_img)
        return {'hr': hr_img.astype(np.float32),
                'lr': self.lr_process(lr_img),
                'fname': f_name}

    def _getitem_test(self, index):
        hr_img, lr_img, f_name = self.getimage(index)
        return {'hr': hr_img.astype(np.float32),
                'lr': self.lr_process(lr_img),
                'fname': f_name}


    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return self.len_data


def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)

def imreader(arg):
    i, obj = arg
    obj.HR_images[i] = obj.imio.read(obj.HR_images[i])
    obj.LR_images[i] = obj.imio.read(obj.LR_images[i])
    # for _ in range(3):
    #     try:
    #         obj.HR_images[i] = obj.imio.read(obj.HR_images[i])
    #         obj.LR_images[i] = obj.imio.read(obj.LR_images[i])
    #         failed = False
    #         break
    #     except:
    #         failed = True
    # if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
    # may use `from multiprocessing import Pool` instead, but less efficient and
    # NOTE: `multiprocessing.Pool` will duplicate given object for each process.
    from multiprocessing.dummy import Pool
    from tqdm import tqdm
    print('Starting to load images via multiple imreaders')
    pool = Pool() # use all threads by default
    for _ in tqdm(pool.imap(imreader, iter_obj(obj.num, obj)), total=obj.num):
        pass
    pool.close()
    pool.join()

if __name__ == '__main__':
    pass
