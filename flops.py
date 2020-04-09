# -*- coding: utf-8 -*-
"""
@author: csmliu
@e-mail: csmliu@outlook.com
"""
import numpy as np

def Conv(in_shape, inc, outc, ks, stride=1, padding=None,
         groups=1, bias=True, mask=None):
    if padding is None:
        padding = ks//2
    if groups != 1:
        assert inc % groups == 0 and outc % groups == 0
        inc = inc // groups
        outc = outc // groups

    _per_pos = ks * ks * inc * outc * groups
    if mask is not None:
        assert all(in_shape == mask.shape)
        n_pos = (mask > 0).sum()
    else:
        n_pos = np.array(in_shape).prod()
    _sum = _per_pos * n_pos
    if bias:
        return _sum + n_pos * outc
    return _sum

def BN(in_shape, inc):
    return np.array(in_shape).prod() * inc * 2 # affine

def ReLU(in_shape, inc):
    return np.array(in_shape).prod() * inc

def pixsf(in_shape, inc, scale):
    _sum_conv = Conv(in_shape, inc, inc*scale**2, 3)
    return np.array(in_shape).prod() * inc + _sum_conv

def pool(in_shape, inc):
    return np.array(in_shape).prod() * inc

def linear(inc, outc, bias=True):
    _sum = inc * outc
    if bias:
        return _sum + outc
    return _sum

def upsample(in_shape, inc, scale=2):
    return (np.array(in_shape)*scale).prod() * inc

def ResBlock(in_shape, inc, mode='CRC', mask=None):
    _sum = 0
    for m in mode:
        if m == 'C':
            _sum += Conv(in_shape, inc, inc, ks=3, mask=mask)
        elif m in 'RPL':
            _sum += ReLU(in_shape, inc)
        elif m == 'B':
            _sum += BN(in_shape, inc)
        else:
            print('mode %s is not supported in ResBlock.'%m)
    return _sum + np.array(in_shape).prod() * inc

def CA(in_shape, inc):
    _sum = np.array(in_shape).prod() * inc # AvgPool
    _sum += linear(inc, inc//16) # 1st conv
    _sum += inc // 16 # ReLU
    _sum += linear(inc//16, inc) # 2nd conv
    _sum += inc // 16 # Sigmoid
    _sum += np.array(in_shape).prod() * inc
    return _sum

def clip(x, layer):
    return np.clip(x, layer, layer+1) - layer

class FLOPs():
    @staticmethod
    def EDSR(in_shape, scale, mask=None, nb=32):
        _sum = 0
        _sum += Conv(in_shape, 3, 256, 3)
        if mask is None:
            _sum += ResBlock(in_shape, 256) * nb
        else:
            for i in range(nb):
                _sum += ResBlock(in_shape, 256, mask=clip(mask, i))
        _sum += Conv(in_shape, 256, 256, 3) + in_shape.prod() * 256
        if scale == 3:
            _sum += pixsf(in_shape, 256, 3)
            in_shape *= 3
        else:
            assert scale in (2, 4)
            for i in range(1, scale, 2):
                _sum += pixsf(in_shape, 256, 2)
                in_shape *= 2
        _sum += Conv(in_shape, 256, 3, 3)
        return _sum

    @staticmethod
    def AdaEDSR(in_shape, scale, mask=None):
        return Conv(in_shape, 256, 128, 3) + \
               Conv(in_shape, 128, 128, 3) * 3 + ReLU(in_shape, 128) * 4 + \
               Conv(in_shape, 128, 1, 3)

    @staticmethod
    def AdaEDSR_fixd(in_shape, scale, mask=None):
        return Conv(in_shape, 256, 128, 3) + \
               Conv(in_shape, 128, 128, 3) * 3 + ReLU(in_shape, 128) * 4 + \
               Conv(in_shape, 128, 1, 3)

    @staticmethod
    def RCAN(in_shape, scale, mask=None):
        _sum = 0
        _sum += Conv(in_shape, 3, 64, 3)
        if mask is None:
            _sum += (ResBlock(in_shape, 64) + CA(in_shape, 64)) * 10 * 20
            _sum += (Conv(in_shape, 64, 64, 3) + in_shape.prod() * 64) * 11
        else:
            for i in range(mask.shape[0]):
                for j in range(20):
                    _sum += ResBlock(in_shape, 64, mask=clip(mask[i], j))
            _sum += CA(in_shape, 64) * 10 * 20
            _sum += (Conv(in_shape, 64, 64, 3) + in_shape.prod() * 64) * 11
        if scale == 3:
            _sum += pixsf(in_shape, 256, 3)
            in_shape *= 3
        else:
            assert scale in (2, 4)
            for i in range(1, scale, 2):
                _sum += pixsf(in_shape, 256, 2)
                in_shape *= 2
        _sum += Conv(in_shape, 64, 3, 3)
        return _sum

    @staticmethod
    def AdaRCAN(in_shape, scale, mask=None):
        return Conv(in_shape, 64, 64, 3) * 4 + ReLU(in_shape, 128) * 4 + \
               Conv(in_shape, 64, 10, 3)


    @staticmethod
    def SRCNN(in_shape, scale, mask=None):
        _sum = 0
        _sum += Conv(in_shape, 1, 64, 9) + ReLU(in_shape, 64)
        _sum += Conv(in_shape, 64, 32, 5) + ReLU(in_shape, 32)
        _sum += Conv(in_shape, 32, 1, 5)
        return _sum

    @staticmethod
    def VDSR(in_shape, scale, mask=None):
        _sum = 0
        _sum += Conv(in_shape, 1, 64, 3) + ReLU(in_shape, 64)
        # NOTE that ReLU is omitted due to that there is no residual
        _sum += ResBlock(in_shape, 64, mode='C') * 18
        _sum += Conv(in_shape, 64, 1, 3)
        _sum += in_shape.prod()
        return _sum

    @staticmethod
    def RDN(in_shape, scale, mask=None):
        def RDB_Conv(in_shape, inc):
            _sum = 0
            _sum += Conv(in_shape, inc, 64, 3) + in_shape.prod() * 64
            _sum += in_shape.prod() * (inc+64)
            return _sum

        def RDB(in_shape):
            _sum = 0
            for i in range(8):
                _sum += RDB_Conv(in_shape, i*64+64)
            _sum += Conv(in_shape, 64*9, 64, 1)
            _sum += in_shape.prod() * 64
            return _sum

        _sum = 0
        _sum += Conv(in_shape, 3, 64, 3)
        _sum += Conv(in_shape, 64, 64, 3)
        _sum += RDB(in_shape) * 16
        _sum += Conv(in_shape, 16*64, 64, 1) + Conv(in_shape, 64, 64, 3)
        _sum += in_shape.prod() * 64
        if scale == 3:
            _sum += pixsf(in_shape, 256, 3)
            in_shape *= 3
        else:
            assert scale in (2, 4)
            for i in range(1, scale, 2):
                _sum += pixsf(in_shape, 256, 2)
                in_shape *= 2
        _sum += Conv(in_shape, 64, 3, 3)
        return _sum

    @staticmethod
    def SAN(in_shape, scale, mask=None):
        def SOCA(in_shape):
            def Covpool(in_shape):
                _sum = 0
                size = in_shape.prod()
                area = size ** 2
                # can be optimized to area + size
                _sum += area * 3
                _sum += size * size * size * 2
                return _sum
            def Sqrtm(in_shape, iterN=5):
                _sum = 0
                ch = 64
                _sum += ch*ch
                _sum += ch*ch*3
                _sum += ch*ch*3
                _sum += (iterN-2)*(ch*ch*5)
                _sum += (ch*ch*5)
                _sum += ch*ch
                return _sum
            _sum = 0
            in_shape = np.min([in_shape, np.array([1000, 1000])], axis=0)
            _sum += Covpool(in_shape)
            in_shape = np.array([64, 64])
            _sum += Sqrtm(in_shape)
            _sum += in_shape.prod()
            in_shape = np.array([1, 1])
            _sum += Conv(in_shape, 64, 64//16, 1)*2 + ReLU(in_shape, 64//16+64)
            return _sum

        def LSRAG(in_shape):
            def RB(in_shape):
                _sum = 0
                _sum += Conv(in_shape, 64, 64, 3) * 2
                _sum += ReLU(in_shape, 64)
                return _sum + in_shape.prod() * 64
            _sum = 0
            _sum += RB(in_shape) * 10
            _sum += SOCA(in_shape)
            _sum += Conv(in_shape, 64, 64, 3)
            return _sum + in_shape.prod()

        def Nonlocal(in_shape):
            def NB(in_shape):
                _sum = 0
                _sum += Conv(in_shape, 64, 32, 1) * 3
                _sum += in_shape.prod()**2 * 32 * 2
                _sum += ReLU(in_shape, 32)
                _sum += Conv(in_shape, 32, 64, 1)
                return _sum
            _sum = 0
            in_shape //= 2
            _sum += NB(in_shape) * 4
            return _sum
        _sum = 0
        _sum += Conv(in_shape, 3, 64, 3)
        _sum += Nonlocal(in_shape) * 2
        _sum += (LSRAG(in_shape) + in_shape.prod()*64) * 20
        if scale == 3:
            _sum += pixsf(in_shape, 256, 3)
            in_shape *= 3
        else:
            assert scale in (2, 4)
            for i in range(1, scale, 2):
                _sum += pixsf(in_shape, 256, 2)
                in_shape *= 2
        # Nonlocal has been calculated before
        _sum += in_shape.prod() * 64
        _sum += Conv(in_shape, 64, 3, 3)
        return _sum


def find(name):
    for func in FLOPs.__dict__.keys():
        if func.lower() == name.lower():
            return func
    raise ValueError('No function named %s is found'%name)

# def cvt(num):
#     units = ['', 'K', 'M', 'G', 'T', 'P', 'Z']
#     cur = 0
#     while num > 1024:
#         cur += 1
#         num /= 1024
#     return '%.3f %s FLOPs' % (num, units[cur])

def cvt(num, binary=True):
    step = 1024 if binary else 1000
    return '%.2f GFLOPs' %(num / step**3)

def chop(input_shape, shave=10, min_size=160000):
    h, w = input_shape
    h_half, w_half = h//2, w//2
    h_size, w_size = h_half+shave, w_half+shave
    if h_size * w_size < min_size:
        return np.array([np.array([h_size, w_size])]*4)
    else:
        ret = np.array([chop(np.array([h_size, w_size]))]*4)
        return ret

def chop_pred(pred, shave=10, min_size=160000):
    if pred is None: return None
    h, w = pred.shape
    h_half, w_half = h//2, w//2
    h_size, w_size = h_half+shave, w_half+shave
    if h_size * w_size < min_size:
        return np.array([
                pred[0:h_size, 0:w_size],
                pred[0:h_size, (w-w_size):w],
                pred[(h-h_size):h, 0:w_size],
                pred[(h-h_size):h, (w-w_size):w]
            ])
    else:
        return np.array([
                chop_pred(pred[0:h_size, 0:w_size]),
                chop_pred(pred[0:h_size, (w-w_size):w]),
                chop_pred(pred[(h-h_size):h, 0:w_size]),
                chop_pred(pred[(h-h_size):h, (w-w_size):w])
            ])


methods = {
    'hr': ['srcnn', 'vdsr'],
    'lr': ['edsr', 'adaedsr', 'adaedsr_fixd', 'rdn', 'rcan', 'san', 'adarcan'],
}
methods = {i:j for j in methods.keys() for i in methods[j]}