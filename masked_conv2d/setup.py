# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, os
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup, find_packages

if torch.cuda.is_available():
    assert torch.matmul(torch.ones(2097153,2).cuda(),torch.ones(2,2).cuda()).min().item()==2, 'Please upgrade from CUDA 9.0 to CUDA 10.0+'

this_dir = os.path.dirname(os.path.realpath(__file__))
torch_dir = os.path.dirname(torch.__file__)
conda_include_dir = '/'.join(torch_dir.split('/')[:-4]) + '/include'

extra = {'cxx': ['-std=c++11', '-fopenmp'], 'nvcc': ['-std=c++11', '-Xcompiler', '-fopenmp']}

setup(
    name='masked_conv2d',
    version='0.1',
    description='Mask Conv',
    packages=find_packages(),
    ext_modules=[
      CUDAExtension('masked_conv2d.masked_conv2d_cuda', [ 'masked_conv2d/src/masked_conv2d_kernel.cu', 'masked_conv2d/src/masked_conv2d_cuda.cpp'],
        #include_dirs=[conda_include_dir],#, this_dir+'/'],
        extra_compile_args=extra)],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

