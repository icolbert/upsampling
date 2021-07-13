import numpy as np

def ceil(x:int, y:int) -> int:
    return int(np.ceil(x / y))


def sub_pixel_convolution_data_reuse_patterns(r, H, C, K, return_total:bool = True):
    M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
    W = pow(r, 2) * pow(K, 2) * pow(C, 2)
    A = (1 + pow(r, 2)) * pow(H, 2) * C 
    P = 2 * pow(r, 2) * pow(H, 2) * C # plus pixel shuffle post-processing
    if not return_total:
        return M, W, A, P
    return M, W, A + P


def NN_resize_convolution_data_reuse_patterns(r, H, C, K, return_total:bool = True):
    M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
    W = pow(K, 2) * pow(C, 2)
    A = 2 * pow(r, 2) * pow(H, 2) * C
    P = (1 + pow(r, 2)) * pow(H, 2) * C # plus nearest neighbor intepolation pre-processing
    if not return_total:
        return M, W, A, P
    return M, W, A + P


def standard_deconvolution_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r, 2) * pow(K, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + K - 1, S = r, P = 1
        M = pow(r + K - 1, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r + K - 1, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")


def fractionally_strided_deconvolution_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = pow(r, 4) * pow(K, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r, 2) * pow(K, 2) * pow(C, 2)
        A = (pow(H + (H - 1)*(r - 1), 2) + pow(r, 2) * pow(H, 2)) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + K - 1, S = r, P = 1
        M = pow(r + K - 1, 2) * pow(r, 2) * pow(H, 2) * pow(C, 2)
        W = pow(r + K - 1, 2) * pow(C, 2)
        A = (pow(H + (H - 1)*(r - 1), 2) + pow(r, 2) * pow(H, 2)) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.") 


def reverse_looping_deconvolution_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r, 2) * pow(K, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + K - 1, S = r, P = 1
        M = pow(r + K - 1, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r + K - 1, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")


def reverse_looping_deconvolution_2_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r, 2) * pow(K, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + K - 1, S = r, P = 1
        M = pow(ceil(r + K - 1, r), 2) * pow(C, 2) * pow(H, 2) * pow(r, 2)
        W = pow(r + K - 1, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")


def transforming_deconvolution_to_convolution_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r, 2) * pow(K, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + K - 1, S = r, P = 1
        M = pow(ceil(r + K - 1, r), 2) * pow(r, 2) * pow(C, 2) * pow(H, 2)
        W = pow(ceil(r + K - 1, r), 2) * pow(r, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")