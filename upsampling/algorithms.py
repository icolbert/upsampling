import torch


def convolution_2d(x:torch.Tensor, weight:torch.Tensor, in_channels:int, out_channels:int, kernel_size:int = 3, padding:int = 0, stride:int = 1):
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