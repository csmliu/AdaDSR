import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import math
import sys
import torch.multiprocessing as mp

from util.util import calc_psnr as calc_psnr
#from util.util import calc_psnr_np as calc_psnr

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset_train = create_dataset('div2k', 'train', opt)
    dataset_size_train = len(dataset_train)
    print('The number of training images = %d' % dataset_size_train)
    dataset_val = create_dataset('div2k', 'val', opt)
    dataset_size_val = len(dataset_val)
    print('The number of val images = %d' % dataset_size_val)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(model.start_epoch + 1, opt.niter + opt.niter_decay + 1):

        # training
        epoch_start_time = time.time()
        epoch_iter = 0
        model.train()
        if hasattr(model, 'depth_gen') and model.depth_gen is not None:
            model.depth_gen.train()

        iter_data_time = iter_start_time = time.time()
        for i, data in enumerate(dataset_train):
            if total_iters % opt.print_freq == 0:
                t_data = time.time() - iter_data_time
            total_iters += 1 #opt.batch_size
            epoch_iter += 1 #opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time)
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data, total_iters)
                # if opt.save_imgs: # Too many images
                #     visualizer.display_current_results(
                #     'train', model.get_current_visuals(), total_iters)
                iter_start_time = time.time()

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d'
                  % (epoch, total_iters))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %.3f sec'
              % (epoch, opt.niter + opt.niter_decay,
                 time.time() - epoch_start_time))
        model.update_learning_rate()

        # val
        if opt.calc_psnr or opt.save_imgs:
            model.eval()
            if hasattr(model, 'depth_gen') and model.depth_gen is not None:
                model.depth_gen.eval() # returns the upper bound of depth
            val_iter_time = time.time()
            tqdm_val = tqdm(dataset_val)
            psnr = [0.0] * dataset_size_val
            time_val = 0
            for i, data in enumerate(tqdm_val):
                model.set_input(data)
                time_val_start = time.time()
                with torch.no_grad():
                    model.test()
                time_val += time.time() - time_val_start
                res = model.get_current_visuals()
                if opt.mode in ('L', 'RGB'):
                    psnr[i] = calc_psnr(res['data_hr'],
                                        res['data_sr'],
                                        opt.scale)
                else: # opt.mode == 'Y':
                    assert opt.mode == 'Y'
                    psnr[i] = calc_psnr(res['data_hr'][:, :1],
                                        res['data_sr'][:, :1],
                                        opt.scale)
                if opt.save_imgs:
                    visualizer.display_current_results('val', res, epoch)
            visualizer.writer.add_scalar('val/psnr', np.mean(psnr), epoch)
            print('End of epoch %d / %d (Val) \t Time Taken: %.3f s \t PSNR: %f'
                % (epoch, opt.niter + opt.niter_decay, time_val, np.mean(psnr)))

        sys.stdout.flush()
