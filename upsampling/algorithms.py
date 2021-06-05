import torch

import numpy as np


def modulo(a:int, b:int) -> int:
    """ modulo operator """
    return a % b


def round(x:torch.Tensor, n_digits:int) -> torch.Tensor:
    return torch.round(x * 10**n_digits) / (10**n_digits)


def convolution_2d(x:torch.Tensor, weight:torch.Tensor, in_channels:int, out_channels:int, kernel_size:int = 3, padding:int = 0, stride:int = 1) -> torch.Tensor:
    """
    2D Convolution - coded for clarity, not for speed
    """
    Ic, Ih, Iw = x.shape
    assert Ih == Iw
    assert Ic == in_channels
    Oh = Ow = 1 + (Ih - kernel_size + 2 * padding) // stride
    output = torch.zeros((out_channels, Oh, Ow))
    for oh in range(Oh):
        for ow in range(Ow):
            for oc in range(out_channels):
                for ic in range(in_channels):
                    for kh in range(kernel_size):
                        for kw in range(kernel_size):
                            ih = stride * oh + kh - padding
                            iw = stride * ow + kw - padding
                            if ih >= 0 and ih < Ih and iw >= 0 and iw < Iw:
                                output[oc,oh,ow] += weight[oc,ic,kh,kw] * x[ic,ih,iw]
    return output


def pixel_shuffle(x:torch.Tensor, scaling_factor:int) -> torch.Tensor:
    """
    Pixel Shuffle - coded for clarity, not speed
    """
    Ic, Ih, Iw = x.shape
    output = torch.zeros((Ic // (scaling_factor**2), Ih * scaling_factor, Iw * scaling_factor))
    Oc, Oh, Ow = output.shape
    for oc in range(Oc):
        for oh in range(Oh):
            for ow in range(Ow):
                ih = int(np.floor(oh / scaling_factor))
                iw = int(np.floor(ow / scaling_factor))
                
                _a = (oh % scaling_factor)
                _b = (ow % scaling_factor)
                _c = oc
                ic = (scaling_factor**2) * _c + (scaling_factor) * _a + _b
                output[oc,oh,ow] = x[ic,ih,iw]
    return output


def weight_shuffle(conv_weights:torch.Tensor, scaling_factor:int) -> torch.Tensor:
    """
    Weight Shuffle - coded for clarity, not speed
    """
    Oc = int(conv_weights.shape[0] / (scaling_factor**2))
    Ic = conv_weights.shape[1]
    K  = conv_weights.shape[2]
    Kh = Kw = scaling_factor * K
    deconv_weights = torch.zeros(Ic, Oc, Kh, Kw)
    for ic_d in range(Ic):
        for oc_d in range(Oc):
            for kh_d in range(Kh):
                for kw_d in range(Kw):
                    kh_c = int(np.floor(kh_d / scaling_factor))
                    kw_c = int(np.floor(kw_d / scaling_factor))
                    ic_c = ic_d
                    _a   = (kh_d % scaling_factor)
                    _b   = (kw_d % scaling_factor)
                    _c   = oc_d
                    oc_c = (scaling_factor**2) * _c + (scaling_factor) * _a + _b
                    deconv_weights[ic_d,oc_d,kh_d,kw_d] = conv_weights[oc_c,ic_c,K - kh_c - 1,K - kw_c - 1]
    return deconv_weights


def weight_convolution(conv_weights:torch.Tensor, in_channels:int, out_channels:int, scaling_factor:int, kernel_size:int = 3):
    """
    Weight Convolution - coded for clarity, not speed

    conv_weights.shape = OC x IC x Kc x Kc, where Kc is the convolution kernel size
    deconv_weights.shape = IC x OC x Kd x Kd, where Kd is the deconvolution kernel size
    """
    z = torch.zeros(in_channels, out_channels, 2 + scaling_factor, 2 + scaling_factor)
    for oc in range(out_channels):
        for ic in range(in_channels):
            for i in range(0, scaling_factor):
                for j in range(0, scaling_factor):
                    z[ic,oc,i:i+kernel_size,j:j+kernel_size] += torch.rot90(conv_weights.data[oc,ic], 2, [0,1])
    return z


def standard_deconvolution(x:torch.Tensor, weight:torch.Tensor, in_channels:int, out_channels:int, kernel_size:int = 3, padding:int = 0, stride:int = 1) -> torch.Tensor:
    """
    Standard Deconvolution (STDD) - coded for clarity, not speed
    """
    Ic, Ih, Iw = x.shape
    assert Ih == Iw
    assert Ic == in_channels
    Oh = Ow = (Ih - 1) * stride - 2 * padding + (kernel_size - 1) + 1
    output = torch.zeros((out_channels, Oh, Ow))
    for oc in range(out_channels):
        for ic in range(in_channels):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    for ih in range(Ih):
                        for iw in range(Iw):
                            oh = (stride * ih) + kh - padding
                            ow = (stride * iw) + kw - padding
                            if oh < Oh and ow < Ow and ow >= 0 and oh >= 0:
                                output[oc,oh,ow] +=  x[ic,ih,iw] * weight[ic,oc,kh,kw]
    return output


def reverse_deconvolution(x:torch.Tensor, weight:torch.Tensor, in_channels:int, out_channels:int, kernel_size:int = 3, padding:int = 0, stride:int = 1) -> torch.Tensor:
    """
    Reverse Deconvolution (REVD) - coded for clarity, not speed
    """
    Ic, Ih, Iw = x.shape
    assert Ih == Iw
    assert Ic == in_channels
    Oh = Ow = (Ih - 1) * stride - 2 * padding + (kernel_size - 1) + 1
    output = torch.zeros((out_channels, Oh, Ow))
    for oc in range(out_channels):
        for ic in range(in_channels):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    for oh_ in range(0, Oh, stride):
                        for ow_ in range(0, Ow, stride):
                            oh = oh_ + modulo(stride - modulo(padding - kh, stride), stride)
                            ow = ow_ + modulo(stride - modulo(padding - kw, stride), stride)
                            ih = (oh + padding - kh) // stride
                            iw = (ow + padding - kw) // stride
                            if ih < Ih and iw < Iw and iw >= 0 and ih >= 0:
                                output[oc,oh,ow] +=  x[ic,ih,iw] * weight[ic,oc,kh,kw]
    return output


def reverse_deconvolution_2(x:torch.Tensor, weight:torch.Tensor, in_channels:int, out_channels:int, kernel_size:int = 3, padding:int = 0, stride:int = 1) -> torch.Tensor:
    """
    Improved Reverse Deconvolution (REVD2) - coded for clarity, not speed
    """
    Ic, Ih, Iw = x.shape
    assert Ih == Iw
    assert Ic == in_channels
    Oh = Ow = (Ih - 1) * stride - 2 * padding + (kernel_size - 1) + 1
    output = torch.zeros((out_channels, Oh, Ow))
    # for the sub-pixel translation, modulo(self.padding, self.stride) = 0
    # this is currently set to generalize for upscaling my non-integer numbers
    kh_offset = KernelOffsetCounter(modulo(padding, stride), stride)
    kw_offset = KernelOffsetCounter(modulo(padding, stride), stride)
    for oc in range(out_channels):
        for oh in range(Oh):
            fh = kh_offset.next()
            for ow in range(Ow):
                fw = kw_offset.next()
                for ic in range(in_channels):
                    for kh_ in range(0, kernel_size, stride):
                        for kw_ in range(0, kernel_size, stride):
                            kh = kh_ + fh
                            kw = kw_ + fw
                            # kh = kh_ + modulo(oh + self.padding, self.stride) # without using the offset counter
                            # kw = kw_ + modulo(ow + self.padding, self.stride) # without using the offset counter
                            ih = (oh + padding - kh) // stride
                            iw = (ow + padding - kw) // stride
                            if ih < Ih and iw < Iw and iw >= 0 and ih >= 0 and \
                                kh < kernel_size and kw < kernel_size:
                                output[oc,oh,ow] +=  x[ic,ih,iw] * weight[ic,oc,kh,kw]
    return output

class KernelOffsetCounter:
    def __init__(self, offset_init:int = 0, cliff:int = 1) -> None:
        self._val = offset_init
        self._cliff = cliff
    
    def next(self) -> int:
        t = self._val
        self._val = (t + 1) if (t + 1) < self._cliff else 0
        return t