import numpy as np
import torch


class Conv2d:
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
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int = 1,
                 padding:int = 0):
        self.weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size)

        self.kernel_size  = kernel_size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.padding      = padding
        self.stride       = stride
    
    def __call__(self, x:torch.tensor) -> torch.tensor:
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
    
class WeightShuffle:
    def __init__(self, scaling_factor:int):
        self.scaling_factor = scaling_factor
    
    def __call__(self, conv_weights:torch.tensor) -> torch.tensor:
        '''
        Assuming the input weights are convolutions and weights are of the size below
        
        convolution.weight.shape = (out_channels, in_channels, kernel_size, kernel_size)
        deconvolution.weight.shape = (in_channels, out_channels, kernel_size, kernel_size)
        '''
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