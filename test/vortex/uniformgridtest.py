import unittest
import itertools
import logging
from ...src.vortex.uniformgrid import UniformGrid
import numpy as np

logger = logging.getLogger(__name__)

class UniformGridTest(unittest.TestCase):


    def test_initialize2x2(self):
        # GIVEN
        grid = UniformGrid(1, np.zeros(3), np.ones(3), False)

        # WHEN
        grid.initialize(0)

        # THEN
        self.assertEquals(8, len(grid.contents))


    def test_insert2x2(self):
        # GIVEN
        grid = UniformGrid(1, np.zeros(3), np.ones(3), False)
        grid.initialize(0)
        
        # WHEN
        grid.insert(np.ones(3), 1)

        # THEN
        offset = grid.offsetFromIndices(np.array((1,1,1)))
        self.assertTrue(np.allclose(1, grid.contents[offset]))

        for coord in itertools.product(range(2), range(2), range(2)):
            if coord != (1, 1, 1):
                offset = grid.offsetFromIndices(np.array(coord))
                self.assertTrue(np.allclose(0, grid.contents[offset]))

    def test_insert3x3(self):
        # GIVEN
        grid = UniformGrid(8, np.zeros(3), np.ones(3), False)
        grid.initialize(0)
        
        # WHEN
        grid.insert(np.ones(3), 1)

        # THEN
        logger.debug(grid.contents)

        offset = grid.offsetFromIndices(np.array((1,1,1)))
        self.assertTrue(np.allclose(1, grid.contents[offset]))

        for coord in itertools.product(range(3), range(3), range(3)):
            if coord != (1, 1, 1):
                offset = grid.offsetFromIndices(np.array(coord))
                self.assertTrue(np.allclose(0, grid.contents[offset]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
