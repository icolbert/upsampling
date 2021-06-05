import torch
import torch.nn as nn

from .enums import DeconvolutionAlgorithms
from .algorithms import *

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
        return convolution_2d(x, weight=self.weight, in_channels=self.in_channels,
            out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding,
                stride=self.stride)

    
class PixelShuffle:
    """
    Pixel Shuffle Layer - Coded for clarity, not speed

    `scaling_factor`:int - the upsampling factor
    """
    def __init__(self, scaling_factor:int):
        self.scaling_factor = scaling_factor
    
    def __call__(self, x:torch.tensor) -> torch.tensor:
        return pixel_shuffle(x, scaling_factor=self.scaling_factor)


class SubPixelConvolution:
    """
    Sub-Pixel Convolution - Coded for clarity, not speed
    """
    def __init__(self,
                 scaling_factor:int,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int = 3,
                 stride:int = 1,
                 padding:int = 1):
        self.convolution = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.pixel_shuffle = PixelShuffle(scaling_factor=scaling_factor)
    
    def __call__(self, x:torch.tensor) -> torch.tensor:
        h = self.convolution(x)
        h = self.pixel_shuffle(h)
        return h


class ResizeConvolution:
    """
    Resize Convolution - Coded for clarity, not speed
    """
    def __init__(self,
                 scaling_factor:int,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int = 3,
                 stride:int = 1,
                 padding:int = 1):
        self.nn_interpolation = nn.Upsample(scale_factor=scaling_factor, mode='nearest')
        self.convolution = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def __call__(self, x:torch.tensor) -> torch.tensor:
        h = self.nn_interpolation(x.unsqueeze(0))
        h = self.convolution(h.squeeze(0))
        return h


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
            return standard_deconvolution(x, weight=self.weight, in_channels=self.in_channels,
                out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding,
                    stride=self.stride)

        elif self.algorithm == DeconvolutionAlgorithms.REVD:
            return reverse_deconvolution(x, weight=self.weight, in_channels=self.in_channels,
                out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding,
                    stride=self.stride)

        elif self.algorithm == DeconvolutionAlgorithms.REVD2:
            return reverse_deconvolution_2(x, weight=self.weight, in_channels=self.in_channels,
                out_channels=self.out_channels, kernel_size=self.kernel_size, padding=self.padding,
                    stride=self.stride)

        else:
            raise Exception(f"{self.algorithm} is not yet supported by this layer.")


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
        return weight_shuffle(conv_weights, scaling_factor=self.scaling_factor)

