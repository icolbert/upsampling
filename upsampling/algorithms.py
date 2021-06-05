import torch

import numpy as np

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


def standard_deconvolution(x:torch.Tensor, weight:torch.Tensor, in_channels:int, out_channels:int, kernel_size:int = 3, padding:int = 0, stride:int = 1) -> torch.Tensor:
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