import math
import numpy as np

def _nearestPowerOf2(x):
    return 1 << (1 - int(math.ceil(x))).bit_length()

__nearestPowerOf2 = np.vectorize(_nearestPowerOf2)

def nearestPowerOf2(x):
	"""Element-wise nearest power of 2 that is greater or equal"""
	return __nearestPowerOf2(x)
