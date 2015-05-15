import unittest
import itertools
import logging
from ...src.vortex.uniformgrid import UniformGrid
import numpy as np

logger = logging.getLogger(__name__)

def gridAsString(grid):
    result = ""
    for z in range(grid.numPoints[2]):
        for y in range(grid.numPoints[1]):
            for x in range(grid.numPoints[0]):
                coord = np.array((x,y,z))
                result += "%f, " % grid.contents[grid.offsetFromIndices(coord)]
            result += "\n"
        result += "\n"
    return result

class UniformGridTest(unittest.TestCase):

    def test_initialize2x2(self):
        # GIVEN
        grid = UniformGrid(1, np.zeros(3), np.ones(3), False)

        # WHEN
        grid.initialize(0)

        # THEN
        self.assertEquals(8, len(grid.contents))


    def test_insert2x2x2Simple(self):
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

    def test_insert2x2x2Interpolated(self):
        # GIVEN
        grid = UniformGrid(1, np.zeros(3), np.ones(3), False)
        grid.initialize(0)
        
        # WHEN
        grid.insert(0.5 * np.ones(3), 1)

        # THEN
        logger.debug("\n" + gridAsString(grid))

        for coord in itertools.product(range(2), range(2), range(2)):
            offset = grid.offsetFromIndices(np.array(coord))
            self.assertTrue(np.allclose(0.125, grid.contents[offset]))

    def test_insert3x3(self):
        # GIVEN
        grid = UniformGrid(8, np.zeros(3), 2*np.ones(3), False)
        grid.initialize(0)
        
        # WHEN
        grid.insert(np.ones(3), 1)

        # THEN
        logger.debug("\n" + gridAsString(grid))

        offset = grid.offsetFromIndices(np.array((1,1,1)))
        self.assertTrue(np.allclose(1, grid.contents[offset]))

        for coord in itertools.product(range(3), range(3), range(3)):
            if coord != (1, 1, 1):
                offset = grid.offsetFromIndices(np.array(coord))
                self.assertTrue(np.allclose(0, grid.contents[offset]))

    def test_insert2x2x3(self):
        # GIVEN
        grid = UniformGrid(2, np.zeros(3), np.array([1, 1, 2]), False)
        grid.initialize(0)
        
        # WHEN
        grid.insert(np.ones(3), 1)

        # THEN
        logger.debug("\n" + gridAsString(grid))

        offset = grid.offsetFromIndices(np.array((1,1,1)))
        self.assertTrue(np.allclose(1, grid.contents[offset]))

        for coord in itertools.product(range(grid.numPoints[0]), range(grid.numPoints[1]), range(grid.numPoints[2])):
            if coord != (1, 1, 1):
                offset = grid.offsetFromIndices(np.array(coord))
                self.assertTrue(np.allclose(0, grid.contents[offset]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
