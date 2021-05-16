import enum
import torch
import numpy as np

def modulo(a:int, b:int) -> int:
    """ modulo operator """
    return a % b


class DeconvolutionAlgorithms(enum.Enum):
    """
    Deconvolution Algorithm Enumerations
    """
    STDD  = 0 # Standard Deconvolution
    STRD  = 1 # Strided Deconvolution
    REVD  = 2 # Reverse Deconvolution
    TDC   = 3 # Transforming Deconvolution to Convolution
    REVD2 = 4 # Reverse Deconvolution-2


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
        Ic, Ih, Iw = x.shape
        assert Ih == Iw
        assert Ic == self.in_channels
        Oh = Ow = 1 + (Ih - self.kernel_size + 2 * self.padding) // self.stride
        output = torch.zeros((self.out_channels, Oh, Ow))
        for oh in range(Oh):
            for ow in range(Ow):
                for oc in range(self.out_channels):
                    for ic in range(self.in_channels):
                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                ih = self.stride * oh + kh - self.padding
                                iw = self.stride * ow + kw - self.padding
                                if ih >= 0 and ih < Ih and iw >= 0 and iw < Iw:
                                    output[oc,oh,ow] += self.weight[oc,ic,kh,kw] * x[ic,ih,iw]
        return output

    
class PixelShuffle:
    """
    Pixel Shuffle Layer - Coded for clarity, not speed

    `scaling_factor`:int - the upsampling factor
    """
    def __init__(self, scaling_factor:int):
        self.scaling_factor = scaling_factor
    
    def __call__(self, x:torch.tensor) -> torch.tensor:
        Ic, Ih, Iw = x.shape
        output = torch.zeros(( Ic//(self.scaling_factor**2), Ih*self.scaling_factor, Iw*self.scaling_factor))
        Oc, Oh, Ow = output.shape
        for oc in range(Oc):
            for oh in range(Oh):
                for ow in range(Ow):
                    ih = int(np.floor(oh / self.scaling_factor))
                    iw = int(np.floor(ow / self.scaling_factor))
                    
                    _a = (oh % self.scaling_factor)
                    _b = (ow % self.scaling_factor)
                    _c = oc
                    ic = (self.scaling_factor**2) * _c + (self.scaling_factor) * _a + _b
                    output[oc,oh,ow] = x[ic,ih,iw]
        return output


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
    
    def __call__(self, x:torch.tensor) -> torch.tensor:

        if self.algorithm == DeconvolutionAlgorithms.STDD:
            return self.standard_deconvolution(x)

        elif self.algorithm == DeconvolutionAlgorithms.STRD:
            raise NotImplementedError("Strided deconvolution (STRD) is not yet implemented.")

        elif self.algorithm == DeconvolutionAlgorithms.REVD:
            return self.reverse_deconvolution(x)

        elif self.algorithm == DeconvolutionAlgorithms.REVD2:
            return self.reverse_deconvolution_2(x)

        elif self.algorithm == DeconvolutionAlgorithms.TDC:
            raise self.transforming_convolution_to_deconvolution(x)

    def standard_deconvolution(self, x:torch.tensor) -> torch.tensor:
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

    def reverse_deconvolution(self, x:torch.tensor) -> torch.tensor:
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

    def reverse_deconvolution_2(self, x:torch.tensor) -> torch.tensor:
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

    def transforming_convolution_to_deconvolution(self, x:torch.tensor) -> torch.tensor:
        Ic, Ih, Iw = x.shape
        assert Ih == Iw
        assert Ic == self.in_channels
        KC = int(np.ceil(self.kernel_size / self.stride)) # size of the convolution kernel
        PK = int(self.stride * KC - self.kernel_size)     # padding of the kernel
        PI = int(KC - 1)                                  # padding of the input

        # augment(pad) kernel and allocate transformed kernels
        K_padded = torch.zeros((self.in_channels, self.out_channels, self.kernel_size + PK, self.kernel_size + PK))
        K_padded[:,:,PK:,PK:] = self.weight # <Ic,Oc,Kh,Kw>
        Kc = torch.zeros((self.in_channels, self.out_channels, self.stride**2, KC, KC)) # <Ic,Oc,S^2,Kh,Kw>

        # pad input
        I_padded = torch.zeros((self.in_channels, Ih + 2 * PI, Iw + 2 * PI))
        I_padded[:,PI: PI + Ih, PI:PI + Iw] = x

        # determine weights of transformed kernels
        for h in range(self.kernel_size + PK):
            for w in range(self.kernel_size + PK):
                for oc in range(self.out_channels):
                    for ic in range(self.in_channels):
                        Kc_h = int(KC - np.ceil((h + 1) / self.stride))
                        Kc_w = int(KC - np.ceil((w + 1) / self.stride))
                        Kc_n = int(self.stride * (h % self.stride) + (w % self.stride))
                        Kc[ic, oc, Kc_n, Kc_h, Kc_w] = K_padded[ic, oc, h, w]

        Oh = Ow = (Ih - 1) * self.stride - 2 * self.padding + (self.kernel_size - 1) + 1
        Th = Tw = Ih + 2 * PI - KC + 1
        output = torch.zeros((self.out_channels, Oh, Ow))
        for oc in range(self.out_channels):
            # Output Split CNN Buffer (unstitched)
            temp_split = np.zeros((Oh // self.stride, Ow // self.stride, self.stride ** 2))
            for ic in range(self.in_channels):
                for os in range(self.stride ** 2): # Parallelize this loop
                    for _oh in range(Th):
                        for _ow in range(Tw):
                            for _kh in range(KC):
                                for _kw in range(KC):
                                    ow = int(_ow * self.stride + (os % self.stride))
                                    oh = int(_oh * self.stride + (os // self.stride))
                                    kh = _oh + _kh 
                                    kw = _ow + _kw
                                    x = I_padded[ic, _kh, _kw]
                                    w = Kc[ic, oc, os, kh, kw]
                                    temp_split[_oh, _ow, os] += x * w
                    # Output pixel look-up for stitching
                    output[oc, oh, ow] += temp_split[_oh, _ow, os]
        return output

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

