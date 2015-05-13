import math
import numpy as np

def _nearestPowerOf2(x):
    return 1 << (1 - int(math.ceil(x))).bit_length()

nearestPowerOf2 = np.vectorize(_nearestPowerOf2(x)
