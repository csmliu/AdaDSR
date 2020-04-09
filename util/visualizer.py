import numpy as np
from os.path import join
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        if opt.isTrain:
            self.name = opt.name
            self.save_dir = join(opt.checkpoints_dir, opt.name, 'log')
            self.writer = SummaryWriter(logdir=join(self.save_dir))
        else:
            self.name = '%s_%s_%d' % (
                opt.name, opt.dataset_name, opt.load_iter)
            self.save_dir = join(opt.checkpoints_dir, opt.name)
            if opt.save_imgs:
                self.writer = SummaryWriter(logdir=join(
                    self.save_dir, 'ckpts', self.name))

    def display_current_results(self, phase, visuals, iters):
        for k, v in visuals.items():
            v = v.cpu()
            if k == 'pred':
                self.process_preds(self.writer, phase, k, v, iters)
            else:
                self.writer.add_image('%s/%s'%(phase, k), v[0]/255, iters)
        self.writer.flush()

    def process_pred(self, pred):
        buffer = BytesIO()
        plt.figure(1)
        plt.clf()
        plt.axis('off')
        img = plt.imshow(pred, cmap=plt.cm.hot)
        plt.colorbar()
        plt.savefig(buffer)
        im = np.array(Image.open(buffer).convert('RGB')).transpose(2, 0, 1)
        buffer.close()
        return im / 255

    def process_preds(self, writer, phase, k, v, iters):
        preds = v[0]
        if len(preds) == 1:
            writer.add_image('%s/%s'%(phase, k),
                             self.process_pred(preds[0]),
                             iters)
        else:
            writer.add_images('%s/%s'%(phase, k),
                              np.stack([self.process_pred(pred)\
                                           for pred in preds]),
                              iters)

    def print_current_losses(self, epoch, iters, losses,
                             t_comp, t_data, total_iters):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' \
                  % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.4e ' % (k, v)
            self.writer.add_scalar('loss/%s'%k, v, total_iters)

        print(message)
