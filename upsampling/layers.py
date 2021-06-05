import torch
import numpy as np

from .enums import DeconvolutionAlgorithms
from .algorithms import *
from .utils import modulo

class Conv2d:
    """
    2-D Convolution Layer - Coded for clarity, not speed

    `in_channels`:int - the number of input channels
    `out_channels`:int - the number of output channels
    `kernel_size`:int - the size of the kernel (assuming a square kernel)
    `stride`:int - kernel stride (default=1)
    `padding`:int - padding of the input feature map (default=0)
    """
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int = 1,
                 padding:int = 0):
        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

        self.kernel_size  = kernel_size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.padding      = padding
        self.stride       = stride
    
    def __call__(self, x:torch.tensor) -> torch.tensor:
        return convolution_2d(x, weight=self.weight, in_channels=self.in_channels, out_channels=self.out_channels,
            kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

    
class PixelShuffle:
    """
    Pixel Shuffle Layer - Coded for clarity, not speed

    `scaling_factor`:int - the upsampling factor
    """
    def __init__(self, scaling_factor:int):
        self.scaling_factor = scaling_factor
    
    def __call__(self, x:torch.tensor) -> torch.tensor:
        return pixel_shuffle(x, scaling_factor=self.scaling_factor)


class Deconvolution:
    """
    Deconvolution Layer - Coded for clarity, not speed

    `in_channels`:int - the number of input channels
    `out_channels`:int - the number of output channels
    `kernel_size`:int - the size of the kernel (assuming a square kernel)
    `stride`:int - kernel stride (default=1)
    `padding`:int - padding of the input feature map (default=0)
    """
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int = 1,
                 padding:int = 0,
                 algorithm:DeconvolutionAlgorithms = DeconvolutionAlgorithms.STDD):
        self.weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size)

        self.kernel_size  = kernel_size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.padding      = padding
        self.stride       = stride
        self.algorithm    = algorithm
    
    def __call__(self, x:torch.Tensor) -> torch.Tensor:

        if self.algorithm == DeconvolutionAlgorithms.STDD:
            return self.standard_deconvolution(x)

        elif self.algorithm == DeconvolutionAlgorithms.STRD:
            return self.strided_deconvolution(x)

        elif self.algorithm == DeconvolutionAlgorithms.REVD:
            return self.reverse_deconvolution(x)

        elif self.algorithm == DeconvolutionAlgorithms.REVD2:
            return self.reverse_deconvolution_2(x)

        elif self.algorithm == DeconvolutionAlgorithms.TDC:
            return self.transforming_convolution_to_deconvolution(x)

    def standard_deconvolution(self, x:torch.Tensor) -> torch.Tensor:
        Ic, Ih, Iw = x.shape
        assert Ih == Iw
        assert Ic == self.in_channels
        Oh = Ow = (Ih - 1) * self.stride - 2 * self.padding + (self.kernel_size - 1) + 1
        output = torch.zeros((self.out_channels, Oh, Ow))
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                for kh in range(self.kernel_size):
                    for kw in range(self.kernel_size):
                        for ih in range(Ih):
                            for iw in range(Iw):
                                oh = (self.stride * ih) + kh - self.padding
                                ow = (self.stride * iw) + kw - self.padding
                                if oh < Oh and ow < Ow and ow >= 0 and oh >= 0:
                                    output[oc,oh,ow] +=  x[ic,ih,iw] * self.weight[ic,oc,kh,kw]
        return output

    def reverse_deconvolution(self, x:torch.Tensor) -> torch.Tensor:
        Ic, Ih, Iw = x.shape
        assert Ih == Iw
        assert Ic == self.in_channels
        Oh = Ow = (Ih - 1) * self.stride - 2 * self.padding + (self.kernel_size - 1) + 1
        output = torch.zeros((self.out_channels, Oh, Ow))
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                for kh in range(self.kernel_size):
                    for kw in range(self.kernel_size):
                        for oh_ in range(0, Oh, self.stride):
                            for ow_ in range(0, Ow, self.stride):
                                oh = oh_ + modulo(self.stride - modulo(self.padding - kh, self.stride), self.stride)
                                ow = ow_ + modulo(self.stride - modulo(self.padding - kw, self.stride), self.stride)
                                ih = (oh + self.padding - kh) // self.stride
                                iw = (ow + self.padding - kw) // self.stride
                                if ih < Ih and iw < Iw and iw >= 0 and ih >= 0:
                                    output[oc,oh,ow] +=  x[ic,ih,iw] * self.weight[ic,oc,kh,kw]
        return output

    def reverse_deconvolution_2(self, x:torch.Tensor) -> torch.Tensor:
        Ic, Ih, Iw = x.shape
        assert Ih == Iw
        assert Ic == self.in_channels
        Oh = Ow = (Ih - 1) * self.stride - 2 * self.padding + (self.kernel_size - 1) + 1
        output = torch.zeros((self.out_channels, Oh, Ow))
        # for image upscaling, modulo(self.padding, self.stride) = 0
        # this is currently set to generalize for upscaling my non-integer numbers
        kh_offset = KernelOffsetCounter(modulo(self.padding, self.stride), self.stride)
        kw_offset = KernelOffsetCounter(modulo(self.padding, self.stride), self.stride)
        for oc in range(self.out_channels):
            for oh in range(Oh):
                fh = kh_offset.next()
                for ow in range(Ow):
                    fw = kw_offset.next()
                    for ic in range(self.in_channels):
                        for kh_ in range(0, self.kernel_size, self.stride):
                            for kw_ in range(0, self.kernel_size, self.stride):
                                kh = kh_ + fh
                                kw = kw_ + fw
                                # kh = kh_ + modulo(oh + self.padding, self.stride)
                                # kw = kw_ + modulo(ow + self.padding, self.stride)
                                ih = (oh + self.padding - kh) // self.stride
                                iw = (ow + self.padding - kw) // self.stride
                                if ih < Ih and iw < Iw and iw >= 0 and ih >= 0 and \
                                    kh < self.kernel_size and kw < self.kernel_size:
                                    output[oc,oh,ow] +=  x[ic,ih,iw] * self.weight[ic,oc,kh,kw]
        return output

    def transforming_convolution_to_deconvolution(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Transforming deconvolution to convolution (TDC) is not yet implemented.")

    def strided_deconvolution(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Strided deconnvolution (STRD) is not yet implemented.")

class KernelOffsetCounter:
    def __init__(self, offset_init:int = 0, cliff:int = 1) -> None:
        self._val = offset_init
        self._cliff = cliff
    
    def next(self) -> int:
        t = self._val
        self._val = (t + 1) if (t + 1) < self._cliff else 0
        return t

class WeightShuffle:
    """
    Weight Shuffle Layer - Coded for clarity, not speed

    Assuming the input weights are convolutions and weights are of the size below (aligned with PyTorch)
      convolution.weight.shape = (out_channels, in_channels, kernel_size, kernel_size)
      deconvolution.weight.shape = (in_channels, out_channels, kernel_size, kernel_size)

    `scaling_factor`:int - the upsampling factor
    """
    def __init__(self, scaling_factor:int):
        self.scaling_factor = scaling_factor
    
    def __call__(self, conv_weights:torch.tensor) -> torch.tensor:
        Oc = int(conv_weights.shape[0] / (self.scaling_factor**2))
        Ic = conv_weights.shape[1]
        K  = conv_weights.shape[2]
        Kh = Kw = self.scaling_factor * K
        deconv_weights = torch.zeros(Ic, Oc, Kh, Kw)
        for ic_d in range(Ic):
            for oc_d in range(Oc):
                for kh_d in range(Kh):
                    for kw_d in range(Kw):
                        kh_c = int(np.floor(kh_d / self.scaling_factor))
                        kw_c = int(np.floor(kw_d / self.scaling_factor))
                        ic_c = ic_d
                        _a   = (kh_d % self.scaling_factor)
                        _b   = (kw_d % self.scaling_factor)
                        _c   = oc_d
                        oc_c = (self.scaling_factor**2) * _c + (self.scaling_factor) * _a + _b
                        deconv_weights[ic_d,oc_d,kh_d,kw_d] = conv_weights[oc_c,ic_c,K - kh_c - 1,K - kw_c - 1]
        return deconv_weights

