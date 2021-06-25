import numpy as np


def sub_pixel_convolution_data_reuse_patterns(r, H, C, K):
    M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
    W = pow(r, 2) * pow(K, 2) * pow(C, 2)
    A = (1 + 3 * pow(r, 2)) * pow(H, 2) * C # plus post-processing
    return M, W, A


def NN_resize_convolution_data_reuse_patterns(r, H, C, K):
    M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
    W = pow(K, 2) * pow(C, 2)
    A = (1 + 3 * pow(r, 2)) * pow(H, 2) * C # plus pre-processing
    return M, W, A


def standard_deconvolution_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r, 2) * pow(K, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + 2, S = r, P = 1
        M = pow(r + 2, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r + 2, 2) * pow(C, 2)
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
        # Kd = r + 2, S = r, P = 1
        M = pow(r + 2, 2) * pow(r, 2) * pow(H, 2) * pow(C, 2)
        W = pow(r + 2, 2) * pow(C, 2)
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
        # Kd = r + 2, S = r, P = 1
        M = pow(r + 2, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r + 2, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")


def reverse_looping_deconvolution_2_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    assert r >= 2, "upsampling factor needs to be >= 2"
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r, 2) * pow(K, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + 2, S = r, P = 1
        M = 4 * pow(r, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r + 2, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")


def transforming_deconvolution_to_convolution_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    assert r >= 2, "upsampling factor needs to be >= 2"
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = pow(r, 2) * pow(K, 2) * pow(C, 2) * pow(H, 2)
        W = pow(r, 2) * pow(K, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + 2, S = r, P = 1
        M = 4 * pow(r, 2) * pow(H, 2) * pow(C, 2)
        W = 4 * pow(r, 2) * pow(C, 2)
        A = (1 + pow(r, 2)) * pow(H, 2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")