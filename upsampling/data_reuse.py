import numpy as np

def reverse_looping_deconvolution_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = (r**2) * (K**2) * (C**2) * (H**2)
        W = (r**2) * (K**2) * (C**2)
        A = (H**2) * C + (r**2) * (H**2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + 2, S = r, P = 1
        M = ((r + 2)**2) * (C**2) * (H**2)
        W = ((r + 2)**2) * (C**2)
        A = (H**2) * C + (r**2) * (H**2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")
    

def fractionally_strided_deconvolution_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = (r**4) * (K**2) * (C**2) * (H**2)
        W = (r**2) * (K**2) * (C**2)
        A = ((H + (H - 1)*(r - 1))**2) * C + (r**2) * (H**2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + 2, S = r, P = 1
        M = ((r + 2)**2) * (C**2) * (H**2) * (r**2)
        W = ((r + 2)**2) * (C**2)
        A = ((H + (H - 1)*(r - 1))**2) * C + (r**2) * (H**2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.") 


def transforming_deconvolution_to_convolution_data_reuse_patterns(r, H, C, K, original_operator="D-SP"):
    if original_operator == "D-SP":
        # Kd = Kc * r, S = r, P = r
        M = (r**2) * (K**2) * (C**2) * (H**2)
        W = (r**2) * (K**2) * (C**2)
        A = (H**2) * C + (r**2) * (H**2) * C
        return M, W, A
    elif original_operator == "D-NN":
        # Kd = r + 2, S = r, P = 1
        M = (r**2) * (np.ceil((r + 2) / r)**2) * (C**2) * (H**2)
        W = (r**2) * (np.ceil((r + 2) / r)**2) * (C**2)
        A = (H**2) * C + (r**2) * (H**2) * C
        return M, W, A
    else:
        raise NotImplementedError(f"{original_operator} is not yet supported.")