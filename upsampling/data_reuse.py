import numpy as np

def ceil(x:int, y:int) -> int:
    return int(np.ceil(x / y))


def sub_pixel_convolution_data_reuse_patterns(upsampling_factor, height, in_channels, kernel_size, return_total:bool = True, width:int = None):
    """
    input:
        upsampling_factor:int - the upsampling factor
        height:int - the height of the input image
        width:int - the width of the input image
        in_channels:int - the number of input channels of the image
        kernel_size:int - the size of the kernel (assuming square)

    output:
        M:int - the compute requirements
        W:int - the weight requirements
        A:int - the activation requirements
        P:int - the post-processing requirements from the pixel shuffle
    """
    width = height if width is None else width
    M = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2) * (height * width)
    W = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2)
    A = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels 
    P = 2 * pow(upsampling_factor, 2) * (height * width) * in_channels # plus pixel shuffle post-processing
    if not return_total:
        return M, W, A, P
    return M, W, A + P


def NN_resize_convolution_data_reuse_patterns(upsampling_factor, height, in_channels, kernel_size, return_total:bool = True, width:int = None):
    """
    input:
        upsampling_factor:int - the upsampling factor
        height:int - the height of the input image
        width:int - the width of the input image
        in_channels:int - the number of input channels of the image
        kernel_size:int - the size of the kernel (assuming square)

    output:
        M:int - the compute requirements
        W:int - the weight requirements
        A:int - the activation requirements
        P:int - the pre-processing requirements from the resize
    """
    width = height if width is None else width
    M = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2) * (height * width)
    W = pow(kernel_size, 2) * pow(in_channels, 2)
    A = 2 * pow(upsampling_factor, 2) * (height * width) * in_channels
    P = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels # plus nearest neighbor intepolation pre-processing
    if not return_total:
        return M, W, A, P
    return M, W, A + P


def standard_deconvolution_data_reuse_patterns(upsampling_factor, height, in_channels, kernel_size, original_operator="D-SP", width:int = None):
    """
    input:
        upsampling_factor:int - the upsampling factor
        height:int - the height of the input image
        width:int - the width of the input image
        in_channels:int - the number of input channels of the image
        kernel_size:int - the size of the kernel (assuming square)

    output:
        M:int - the compute requirements
        W:int - the weight requirements
        A:int - the activation requirements
    """
    width = height if width is None else width
    if original_operator == "D-SP":
        # Kd = Kc * upsampling_factor, S = upsampling_factor, P = upsampling_factor
        M = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2) * (height * width)
        W = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2)
        A = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = upsampling_factor + kernel_size - 1, S = upsampling_factor, P = 1
        M = pow(upsampling_factor + kernel_size - 1, 2) * pow(in_channels, 2) * (height * width)
        W = pow(upsampling_factor + kernel_size - 1, 2) * pow(in_channels, 2)
        A = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")


def fractionally_strided_deconvolution_data_reuse_patterns(upsampling_factor, height, in_channels, kernel_size, original_operator="D-SP", width:int = None):
    """
    input:
        upsampling_factor:int - the upsampling factor
        height:int - the height of the input image
        width:int - the width of the input image
        in_channels:int - the number of input channels of the image
        kernel_size:int - the size of the kernel (assuming square)

    output:
        M:int - the compute requirements
        W:int - the weight requirements
        A:int - the activation requirements
    """
    width = height if width is None else width
    if original_operator == "D-SP":
        # Kd = Kc * upsampling_factor, S = upsampling_factor, P = upsampling_factor
        M = pow(upsampling_factor, 4) * pow(kernel_size, 2) * pow(in_channels, 2) * (height * width)
        W = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2)
        A = (pow(height + (height - 1)*(upsampling_factor - 1), 2) + pow(upsampling_factor, 2) * (height * width)) * in_channels
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = upsampling_factor + kernel_size - 1, S = upsampling_factor, P = 1
        M = pow(upsampling_factor + kernel_size - 1, 2) * pow(upsampling_factor, 2) * (height * width) * pow(in_channels, 2)
        W = pow(upsampling_factor + kernel_size - 1, 2) * pow(in_channels, 2)
        A = (pow(height + (height - 1)*(upsampling_factor - 1), 2) + pow(upsampling_factor, 2) * (height * width)) * in_channels
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.") 


def reverse_looping_deconvolution_data_reuse_patterns(upsampling_factor, height, in_channels, kernel_size, original_operator="D-SP", width:int = None):
    """
    input:
        upsampling_factor:int - the upsampling factor
        height:int - the height of the input image
        width:int - the width of the input image
        in_channels:int - the number of input channels of the image
        kernel_size:int - the size of the kernel (assuming square)

    output:
        M:int - the compute requirements
        W:int - the weight requirements
        A:int - the activation requirements
    """
    width = height if width is None else width
    if original_operator == "D-SP":
        # Kd = Kc * upsampling_factor, S = upsampling_factor, P = upsampling_factor
        M = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2) * (height * width)
        W = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2)
        A = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = upsampling_factor + kernel_size - 1, S = upsampling_factor, P = 1
        M = pow(upsampling_factor + kernel_size - 1, 2) * pow(in_channels, 2) * (height * width)
        W = pow(upsampling_factor + kernel_size - 1, 2) * pow(in_channels, 2)
        A = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")


def reverse_looping_deconvolution_2_data_reuse_patterns(upsampling_factor, height, in_channels, kernel_size, original_operator="D-SP", width:int = None):
    """
    input:
        upsampling_factor:int - the upsampling factor
        height:int - the height of the input image
        width:int - the width of the input image
        in_channels:int - the number of input channels of the image
        kernel_size:int - the size of the kernel (assuming square)

    output:
        M:int - the compute requirements
        W:int - the weight requirements
        A:int - the activation requirements
    """
    width = height if width is None else width
    if original_operator == "D-SP":
        # Kd = Kc * upsampling_factor, S = upsampling_factor, P = upsampling_factor
        M = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2) * (height * width)
        W = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2)
        A = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = upsampling_factor + kernel_size - 1, S = upsampling_factor, P = 1
        M = pow(ceil(upsampling_factor + kernel_size - 1, upsampling_factor), 2) * pow(in_channels, 2) * (height * width) * pow(upsampling_factor, 2)
        W = pow(upsampling_factor + kernel_size - 1, 2) * pow(in_channels, 2)
        A = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")


def transforming_deconvolution_to_convolution_data_reuse_patterns(upsampling_factor, height, in_channels, kernel_size, original_operator="D-SP", width:int = None):
    """
    input:
        upsampling_factor:int - the upsampling factor
        height:int - the height of the input image
        width:int - the width of the input image
        in_channels:int - the number of input channels of the image
        kernel_size:int - the size of the kernel (assuming square)

    output:
        M:int - the compute requirements
        W:int - the weight requirements
        A:int - the activation requirements
    """
    width = height if width is None else width
    if original_operator == "D-SP":
        # Kd = Kc * upsampling_factor, S = upsampling_factor, P = upsampling_factor
        M = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2) * (height * width)
        W = pow(upsampling_factor, 2) * pow(kernel_size, 2) * pow(in_channels, 2)
        A = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = upsampling_factor + kernel_size - 1, S = upsampling_factor, P = 1
        M = pow(ceil(upsampling_factor + kernel_size - 1, upsampling_factor), 2) * pow(upsampling_factor, 2) * pow(in_channels, 2) * (height * width)
        W = pow(ceil(upsampling_factor + kernel_size - 1, upsampling_factor), 2) * pow(upsampling_factor, 2) * pow(in_channels, 2)
        A = (1 + pow(upsampling_factor, 2)) * (height * width) * in_channels
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")