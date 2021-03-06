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
        grid.initialize(0.0)
        
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
        grid.initialize(0.0)
        
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
        grid.initialize(0.0)
        
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
        grid.initialize(0.0)
        
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


    def test_interpolate2x2(self):
        # GIVEN
        grid = UniformGrid(1, np.zeros(3), np.ones(3), False)
        grid.initialize(0.0)
        grid.insert(np.ones(3), 1.0)

        # THEN
        for coord in itertools.product(range(grid.numPoints[0]), range(grid.numPoints[1]), range(grid.numPoints[2])):
            npCoord = np.array(coord)
            if coord == (1, 1, 1):
                self.assertTrue(np.allclose(1, grid.interpolate(npCoord)))
            else:
                self.assertTrue(np.allclose(0, grid.interpolate(npCoord)))

        self.assertTrue(np.allclose(0.5, grid.interpolate(np.array([0.5, 1, 1]))))
        self.assertTrue(np.allclose(0.5, grid.interpolate(np.array([1, 0.5, 1]))))
        self.assertTrue(np.allclose(0.5, grid.interpolate(np.array([1, 1, 0.5]))))
        self.assertTrue(np.allclose(0.25, grid.interpolate(np.array([0.5, 0.5, 1]))))
        self.assertTrue(np.allclose(0.25, grid.interpolate(np.array([0.5, 1, 0.5]))))
        self.assertTrue(np.allclose(0.25, grid.interpolate(np.array([1, 0.5, 0.5]))))
        self.assertTrue(np.allclose(0.125, grid.interpolate(np.array([0.5, 0.5, 0.5]))))

    def test_interpolate3x3(self):
        # GIVEN
        grid = UniformGrid(8, np.zeros(3), 2*np.ones(3), False)
        grid.initialize(0)
        grid.insert(np.zeros(3), 1)

        # THEN
        for coord in itertools.product(range(grid.numPoints[0]), range(grid.numPoints[1]), range(grid.numPoints[2])):
            npCoord = np.array(coord)
            if coord == (0, 0, 0):
                self.assertTrue(np.allclose(1, grid.interpolate(npCoord)))
            else:
                self.assertTrue(np.allclose(0, grid.interpolate(npCoord)))

        self.assertTrue(np.allclose(0, grid.interpolate(np.array([1.5, 1.5, 1.5]))))

    def test_interpolate2x2NonZeroMinCorner(self):
        # GIVEN
        grid = UniformGrid(1, np.ones(3), 2*np.ones(3), False)
        grid.initialize(0)
        grid.insert(np.ones(3), 1)

        # THEN
        for coord in itertools.product(range(1, grid.numPoints[0] + 1), range(1, grid.numPoints[1] + 1), range(1, grid.numPoints[2] + 1)):
            npCoord = np.array(coord)
            if coord == (1, 1, 1):
                self.assertTrue(np.allclose(1, grid.interpolate(npCoord)))
            else:
                self.assertTrue(np.allclose(0, grid.interpolate(npCoord)))

        self.assertTrue(np.allclose(0.5, grid.interpolate(np.array([1.5, 1, 1]))))
        self.assertTrue(np.allclose(0.5, grid.interpolate(np.array([1, 1.5, 1]))))
        self.assertTrue(np.allclose(0.5, grid.interpolate(np.array([1, 1, 1.5]))))
        self.assertTrue(np.allclose(0.25, grid.interpolate(np.array([1.5, 1.5, 1]))))
        self.assertTrue(np.allclose(0.25, grid.interpolate(np.array([1.5, 1, 1.5]))))
        self.assertTrue(np.allclose(0.25, grid.interpolate(np.array([1, 1.5, 1.5]))))
        self.assertTrue(np.allclose(0.125, grid.interpolate(np.array([1.5, 1.5, 1.5]))))

    def test_interpolate2x2Vector(self):
        # GIVEN
        grid = UniformGrid(1, np.zeros(3), np.ones(3), False)
        grid.initialize(np.zeros(3))
        grid.insert(np.ones(3), np.ones(3))

        # THEN
        for coord in itertools.product(range(grid.numPoints[0]), range(grid.numPoints[1]), range(grid.numPoints[2])):
            npCoord = np.array(coord)
            if coord == (1, 1, 1):
                self.assertTrue(np.allclose(np.ones(3), grid.interpolate(npCoord)))
            else:
                self.assertTrue(np.allclose(np.zeros(3), grid.interpolate(npCoord)))

        self.assertTrue(np.allclose(0.5*np.ones(3), grid.interpolate(np.array([0.5, 1, 1]))))
        self.assertTrue(np.allclose(0.5*np.ones(3), grid.interpolate(np.array([1, 0.5, 1]))))
        self.assertTrue(np.allclose(0.5*np.ones(3), grid.interpolate(np.array([1, 1, 0.5]))))
        self.assertTrue(np.allclose(0.25*np.ones(3), grid.interpolate(np.array([0.5, 0.5, 1]))))
        self.assertTrue(np.allclose(0.25*np.ones(3), grid.interpolate(np.array([0.5, 1, 0.5]))))
        self.assertTrue(np.allclose(0.25*np.ones(3), grid.interpolate(np.array([1, 0.5, 0.5]))))
        self.assertTrue(np.allclose(0.125*np.ones(3), grid.interpolate(np.array([0.5, 0.5, 0.5]))))


    def test_decimateSingleCellUnitDecimation(self):
        # GIVEN
        srcGrid = UniformGrid(1, np.zeros(3), np.ones(3), False)
        decGrid = UniformGrid()

        # WHEN
        decGrid.decimate(srcGrid, 1)

        # THEN
        self.assertTrue(np.allclose(srcGrid.gridExtent, decGrid.gridExtent))
        self.assertTrue(np.allclose(srcGrid.minCorner, decGrid.minCorner))
        self.assertTrue(np.allclose(srcGrid.numPoints , decGrid.numPoints))
        self.assertTrue(np.allclose(srcGrid.cellExtent, decGrid.cellExtent))
        self.assertTrue(np.allclose(srcGrid.cellsPerExtent, decGrid.cellsPerExtent))

    def test_decimate2x2x2CellTwoDecimation(self):
        # GIVEN
        srcGrid = UniformGrid(8, np.zeros(3), 2*np.ones(3), False)
        decGrid = UniformGrid()

        # WHEN
        decGrid.decimate(srcGrid, 2)

        # THEN
        self.assertTrue(np.allclose(srcGrid.gridExtent, decGrid.gridExtent))
        self.assertTrue(np.allclose(srcGrid.minCorner, decGrid.minCorner))
        self.assertTrue(np.allclose(2*np.ones(3), decGrid.numPoints))
        self.assertTrue(np.allclose(2*np.ones(3), decGrid.cellExtent))
        self.assertTrue(np.allclose(0.5*np.ones(3), decGrid.cellsPerExtent))

    def test_getitem(self):
        # GIVEN
        grid = UniformGrid(8, np.zeros(3), 2*np.ones(3), False)
        grid.initialize(0.0)
        grid.insert(np.ones(3), 1)

        # THEN
        self.assertTrue(np.allclose(0, grid[grid.offsetFromIndices(np.array([0,0,0]))]))
        self.assertTrue(np.allclose(0, grid[grid.offsetFromIndices(np.array([1,0,0]))]))
        self.assertTrue(np.allclose(0, grid[grid.offsetFromIndices(np.array([0,1,0]))]))
        self.assertTrue(np.allclose(0, grid[grid.offsetFromIndices(np.array([0,0,1]))]))
        self.assertTrue(np.allclose(0, grid[grid.offsetFromIndices(np.array([1,1,0]))]))
        self.assertTrue(np.allclose(0, grid[grid.offsetFromIndices(np.array([0,1,1]))]))
        self.assertTrue(np.allclose(0, grid[grid.offsetFromIndices(np.array([1,0,1]))]))
        self.assertTrue(np.allclose(1, grid[grid.offsetFromIndices(np.array([1,1,1]))]))

    def test_setitem(self):
        # GIVEN
        grid = UniformGrid(8, np.zeros(3), 2*np.ones(3), False)
        grid.initialize(0.0)

        # WHEN
        grid[0] = 1
        grid[1] = 2

        # THEN
        self.assertEqual(1, grid[0])
        self.assertEqual(2, grid[1])

if __name__ == '__main__':
    unittest.main()
