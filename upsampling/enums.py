import enum

class DeconvolutionAlgorithms(enum.Enum):
    """
    Deconvolution Algorithm Enumerations
    """
    STDD  = 0 # Standard Deconvolution
    STRD  = 1 # Strided Deconvolution
    REVD  = 2 # Reverse Deconvolution
    TDC   = 3 # Transforming Deconvolution to Convolution
    REVD2 = 4 # Reverse Deconvolution-2
