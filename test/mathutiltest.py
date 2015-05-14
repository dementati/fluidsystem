import unittest
import numpy as np

from ..src import mathutil as mu

class MathUtilTest(unittest.TestCase):

	def test_nearestPowerOf2(self):
		self.npAssertEquals(np.array(0), mu.nearestPowerOf2(0))
		self.npAssertEquals(np.array(2), mu.nearestPowerOf2(np.finfo(float).tiny))
		self.npAssertEquals(np.array(2), mu.nearestPowerOf2(2 - np.finfo(float).eps))
		self.npAssertEquals(np.array(2), mu.nearestPowerOf2(2))

	def npAssertEquals(self, np1, np2):
		return np.all(np1 == np2)

if __name__ == '__main__':
    unittest.main()
