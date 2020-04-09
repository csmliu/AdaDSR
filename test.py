import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model, networks as N
from util.visualizer import Visualizer
from tqdm import tqdm
from train import calc_psnr
import time
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict as odict
from copy import deepcopy
import shutil

# for FLOPs
from flops import FLOPs, find, methods, chop, chop_pred, cvt

if __name__ == '__main__':
    opt = TestOptions().parse()
    # log_dir = '%s/%s/psnr_x%s.txt' % (opt.checkpoints_dir, opt.name, opt.scale)
    # f = open(log_dir, 'a')

    opt_depths = deepcopy(opt.depth)
    opt.depth = [opt.depth[0]]
    if not isinstance(opt.load_iter, list):
        load_iters = [opt.load_iter]
    else:
        load_iters = deepcopy(opt.load_iter)

    if not isinstance(opt.dataset_name, list):
        dataset_names = [opt.dataset_name]
    else:
        dataset_names = deepcopy(opt.dataset_name)
    datasets = odict()
    for dataset_name in dataset_names:
        dataset = create_dataset(dataset_name, 'test', opt)
        datasets[dataset_name] = tqdm(dataset)

    # FLOPs
    if opt.model in ('adaedsr', 'adarcan'):
        func = getattr(FLOPs, find(opt.model[3:]))
    elif opt.model == 'adaedsr_fixd':
        func = FLOPs.EDSR
    else:
        func = getattr(FLOPs, find(opt.model))

    for load_iter in load_iters:
        opt.load_iter = load_iter
        model = create_model(opt)
        model.setup(opt)
        model.eval()
        with_depth = hasattr(model, 'nc_adapter') and model.nc_adapter
        log_dir = '%s/%s/logs/log_x%d_epoch%d.txt' % (
                opt.checkpoints_dir, opt.name, opt.scale, load_iter)
        os.makedirs(os.path.split(log_dir)[0], exist_ok=True)
        f = open(log_dir, 'a')

        for depth in opt_depths:
            if with_depth:
                opt.depth = [depth]
                model.depth_gen = N.num_generator(opt.depth)

            for dataset_name in dataset_names:
                opt.dataset_name = dataset_name
                tqdm_val = datasets[dataset_name]
                dataset_test = tqdm_val.iterable
                dataset_size_test = len(dataset_test)

                print('='*80)
                print(dataset_name, depth)
                tqdm_val.reset()


                if opt.matlab:
                    shutil.rmtree('./tmp', ignore_errors=True)
                    os.makedirs('./tmp/HR', exist_ok=True)
                    os.makedirs('./tmp/SR', exist_ok=True)

                psnr = [0.0] * dataset_size_test
                ssim = [0.0] * dataset_size_test
                _sum = [0.0] * dataset_size_test # FLOPs
                if with_depth:
                    depths = [0.0] * dataset_size_test
                time_val = 0
                for i, data in enumerate(tqdm_val):
                    if not opt.FLOPs_only or opt.model not in (
                            'srcnn', 'vdsr', 'rdn', 'san', 'edsr', 'rcan'):
                        torch.cuda.empty_cache()
                        model.set_input(data)
                        torch.cuda.synchronize()
                        time_val_start = time.time()
                        model.test(opt.FLOPs_only)
                        torch.cuda.synchronize()
                        time_val += time.time() - time_val_start
                        res = model.get_current_visuals()
                        if with_depth:
                            depths[i] = (torch.ceil(torch.clamp(
                                res['pred'], 0, opt.n_resblocks)).mean()).item()
                        if not opt.matlab:
                            if opt.mode in ('L', 'RGB'):
                                psnr[i] = calc_psnr(res['data_hr'],
                                                    res['data_sr'],
                                                    opt.scale)
                            else: # opt.mode == 'Y':
                                assert opt.mode == 'Y'
                                psnr[i] = calc_psnr(res['data_hr'][:, :1],
                                                    res['data_sr'][:, :1],
                                                    opt.scale)
                    # FLOPs
                    in_shape = np.array(data[methods[opt.model]].shape[-2:])
                    scale = opt.scale
                    if with_depth:
                        mask = np.array(res['pred'].cpu().squeeze())
                    else:
                        mask = None
                    if opt.chop:
                        in_shapes = chop(in_shape)
                        if mask is not None:
                            if len(mask.shape) == 2:
                                masks = chop_pred(mask)
                            elif len(mask.shape) == 3:
                                masks = np.array([chop_pred(m) for m in mask])
                                masks = masks.transpose(1, 0, 2, 3)
                            else:
                                raise ValueError
                        for ii in range(in_shapes.shape[0]):
                            maskii = masks[ii] if mask is not None else None
                            _sum[i] += func(in_shapes[ii], scale, maskii)
                        if opt.model in ('adarcan', 'adaedsr', 'adaedsr_fixd'):
                            _sum[i] += getattr(FLOPs,
                                    find(opt.model))(in_shape, scale)
                    else:
                        _sum[i] = func(in_shape, scale, mask)
                    if opt.FLOPs_only:
                        continue
                    if opt.save_imgs:
                        folder_dir = '%s/compare/x%d/%s/%s' % (
                            opt.checkpoints_dir,
                            opt.scale,
                            opt.dataset_name,
                            os.path.basename(data['fname'][0]).split('.')[0])
                        depth_folder_dir = folder_dir+'_depth'
                        os.makedirs(depth_folder_dir, exist_ok=True)
                        if with_depth:
                            save_dir = '%s/%s_%ddepth.png' % (
                                    folder_dir, opt.name, depth)
                            for idx in range(res['pred'].shape[1]):
                                pred_dir = '%s/%s_d%d_p%d' % (
                                        depth_folder_dir, opt.name, depth, idx)
                                plt.figure(1)
                                plt.clf()
                                plt.axis('off')
                                img = plt.imshow(res['pred'][0, idx].cpu(),
                                                vmin=0, vmax=opt.n_resblocks,
                                                cmap=plt.cm.hot)
                                plt.colorbar()
                                plt.savefig(pred_dir)
                        else:
                            save_dir = '%s/%s.png' % (folder_dir, opt.name)
                        dataset_test.imio.write(np.array(res['data_sr'][0].cpu()
                                ).astype(np.uint8), save_dir)
                    if opt.matlab:
                        dataset_test.imio.write(np.array(res['data_sr'][0][:,
                            opt.scale:-opt.scale, opt.scale:-opt.scale].cpu()
                            ).astype(np.uint8), './tmp/SR/%d.png'%i)
                        dataset_test.imio.write(np.array(res['data_hr'][0][:,
                            opt.scale:-opt.scale, opt.scale:-opt.scale].cpu()
                            ).astype(np.uint8), './tmp/HR/%d.png'%i)
                if opt.FLOPs_only:
                    print('dataset: %s, depth: %d\n%s %s' % (
                        dataset_name, depth,
                        cvt(np.sum(_sum)), cvt(np.mean(_sum))))
                    f.write('dataset: %s, depth: %d\n%s %s\n' % (
                        dataset_name, depth,
                        cvt(np.sum(_sum)), cvt(np.mean(_sum))))
                    f.flush()
                    continue

                if opt.matlab:
                    print('Calcualting PSNR and SSIM with matlab ...')
                    os.system('matlab -nodesktop -nosplash -r'
                            ' "run(\'calc_psnr_ssim.m\');exit;"'
                            ' > /dev/null')
                    fres = open('result.txt', 'r')
                    m_psnr, m_ssim = fres.readlines()[0].strip().split()
                    fres.close()
                    avg_psnr, avg_ssim = m_psnr, m_ssim
                else:
                    avg_psnr = '%.6f'%np.mean(psnr)
                    avg_ssim = '%.6f'%np.mean(ssim)

                if with_depth:
                    print('desired depth:', depth,
                          'mean depth:', np.mean(depths))
                    f.write('dataset: %s, depth: %d, mean_depth: %.4f, '
                            'PSNR: %s, SSIM: %s, Time: %.3f sec.\n%s %s\n'
                            % (dataset_name, depth, np.mean(depths),
                               avg_psnr, avg_ssim, time_val,
                               cvt(np.sum(_sum)), cvt(np.mean(_sum))))
                else:
                    f.write('dataset: %s, PSNR: %s, SSIM: %s, '
                            'Time: %.3f sec.\n%s %s\n'
                            % (dataset_name, avg_psnr, avg_ssim, time_val,
                               cvt(np.sum(_sum)), cvt(np.mean(_sum))))
                print('Time: %.3f s AVG Time: %.3f ms PSNR: %s SSIM: %s\n%s %s'
                      % (time_val, time_val/dataset_size_test*1000, avg_psnr,
                         avg_ssim, cvt(np.sum(_sum)), cvt(np.mean(_sum))))
                f.flush()
            f.write('\n')
        f.close()
    for dataset in datasets:
        datasets[dataset].close()
