import math
import numpy as np

def _nearestPowerOf2(x):
    assert x >= 0, "x must be non-negative"
    
    if x == 0:
        return 0

    if x <= 1:
        return 2

    return 1 << (1 - int(math.ceil(x))).bit_length()

__nearestPowerOf2 = np.vectorize(_nearestPowerOf2)

def nearestPowerOf2(x):
    """Element-wise nearest power of 2 that is greater or equal"""
    assert isinstance(x, np.ndarray), "x must be a numpy array"
    return __nearestPowerOf2(x)
