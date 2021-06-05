import torch
import numpy as np

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

