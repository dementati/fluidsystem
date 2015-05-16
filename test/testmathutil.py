import unittest
import numpy as np

from ..src import mathutil as mu

class TestMathUtil(unittest.TestCase):

    def test_nearestPowerOf2(self):
        self.assertTrue(np.array(0) == mu.nearestPowerOf2(np.array(0)))
        self.assertTrue(np.array(1) == mu.nearestPowerOf2(np.array(0.1)))
        self.assertTrue(np.array(1) == mu.nearestPowerOf2(np.array(1)))
        self.assertTrue(np.array(2) == mu.nearestPowerOf2(np.array(1.1)))
        self.assertTrue(np.array(2) == mu.nearestPowerOf2(np.array(2)))
        self.assertTrue(np.array(4) == mu.nearestPowerOf2(np.array(2.1)))
        self.assertTrue(np.array(4) == mu.nearestPowerOf2(np.array(4)))
        self.assertTrue(np.array(8) == mu.nearestPowerOf2(np.array(4.1)))
        self.assertTrue(np.array(8) == mu.nearestPowerOf2(np.array(8)))
        self.assertTrue(np.array(16) == mu.nearestPowerOf2(np.array(8.1)))
        self.assertTrue(np.all(np.array([1]) == mu.nearestPowerOf2(np.array([0.1]))))
        self.assertTrue(np.all(np.array([0, 1, 1, 2, 2, 4, 4, 8, 8, 16]) == mu.nearestPowerOf2(np.array([0, 0.1, 1, 1.1, 2, 2.1, 4, 4.1, 8, 8.1]))))

if __name__ == '__main__':
    unittest.main()
