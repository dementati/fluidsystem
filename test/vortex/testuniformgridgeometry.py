from __future__ import division

import unittest
from ...src.vortex.uniformgridgeometry import UniformGridGeometry
import numpy as np
import logging
import itertools

logger = logging.getLogger(__name__)

class UniformGridGeometryTest(unittest.TestCase):

    def test_constructSingleElement1x1x1(self):
        # WHEN
        gridGeometry = UniformGridGeometry(1, np.zeros(3), np.ones(3), False)

        # THEN
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(2*np.ones(3), gridGeometry.numPoints))
        self.assertEquals(8, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.getNumCells()))

    def test_construct8Elements2x2x2(self):
        # WHEN
        gridGeometry = UniformGridGeometry(8, np.zeros(3), 2*np.ones(3), False)

        # THEN
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(2*np.ones(3), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(3*np.ones(3), gridGeometry.numPoints))
        self.assertEquals(27, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(2*np.ones(3), gridGeometry.getNumCells()))

    def test_construct4Elements2x2(self):
        # WHEN 
        gridGeometry = UniformGridGeometry(4, np.zeros(2), 2*np.ones(2), False)

        # THEN
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.array([2, 2, 0]), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.array([1, 1, 0]), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.array([1, 1, 1.0/np.finfo(float).tiny]), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(np.array([3, 3, 2]), gridGeometry.numPoints))
        self.assertEquals(18, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.array([2, 2, 1]), gridGeometry.getNumCells()))

    def test_constructSingleElement1x1x1ToPowerOf2(self):
        # WHEN
        gridGeometry = UniformGridGeometry(1, np.zeros(3), np.ones(3), True)

        # THEN 
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(2*np.ones(3), gridGeometry.numPoints))
        self.assertEquals(8, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.getNumCells()))

    def test_constructTwoElements1x1x1(self):
        # WHEN
        gridGeometry = UniformGridGeometry(2, np.zeros(3), np.ones(3), False)

        # THEN
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(2*np.ones(3), gridGeometry.numPoints))
        self.assertEquals(8, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.getNumCells()))


    def test_construct2Elements1x1x1ToPowerOf2(self):
        # WHEN
        gridGeometry = UniformGridGeometry(2, np.zeros(3), np.ones(3), True)

        # THEN
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(2*np.ones(3), gridGeometry.numPoints))
        self.assertEquals(8, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.getNumCells()))

    def test_construct2Elements2x1x1(self):
        # WHEN
        gridGeometry = UniformGridGeometry(2, np.zeros(3), np.array([2, 1, 1]), False)

        # THEN 
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.array([2,1,1]), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(np.array([3, 2, 2]), gridGeometry.numPoints))
        self.assertEquals(12, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.array([2, 1, 1]), gridGeometry.getNumCells()))

    def test_construct6Elements3x2x1ToPowerOf2(self):
        # WHEN
        gridGeometry = UniformGridGeometry(6, np.zeros(3), np.array([3, 2, 1]), True)

        # THEN 
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.array([3, 2, 1]), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.array([3/4, 1, 1]), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.array([4/3, 1, 1]), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(np.array([5, 3, 2]), gridGeometry.numPoints))
        self.assertEquals(30, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.array([4, 2, 1]), gridGeometry.getNumCells()))

    def test_constructSingleElement2x1x1(self):
        # WHEN
        gridGeometry = UniformGridGeometry(1, np.zeros(3), np.array([2, 1, 1]), True)

        # THEN
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.array([2, 1, 1]), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.array([1, 1, 1]), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.array([1, 1, 1]), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(np.array([3, 2, 2]), gridGeometry.numPoints))
        self.assertEquals(12, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.array([2, 1, 1]), gridGeometry.getNumCells()))

    def test_constructSingleElement30x1x1(self):
        # WHEN
        gridGeometry = UniformGridGeometry(1, np.zeros(3), np.array([30, 1, 1]), False)

        # THEN
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.array([30, 1, 1]), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.array([6, 1, 1]), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.array([1/6, 1, 1]), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(np.array([6, 2, 2]), gridGeometry.numPoints))
        self.assertEquals(24, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.array([5, 1, 1]), gridGeometry.getNumCells()))

    def test_constructSingleElement1x1x1NonZeroMinCorner(self):
        # WHEN
        gridGeometry = UniformGridGeometry(1, np.ones(3), 2*np.ones(3), False)

        # THEN
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.minCorner))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.gridExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellExtent))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.cellsPerExtent))
        self.assertTrue(np.allclose(2*np.ones(3), gridGeometry.numPoints))
        self.assertEquals(8, gridGeometry.getGridCapacity())
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.getNumCells()))

    def test_indicesOfPosition(self):
        # GIVEN 
        gridGeometry = UniformGridGeometry(8, np.zeros(3), 2*np.ones(3), False)

        # THEN
        self.assertTrue(np.all(np.zeros(3) == gridGeometry.indicesOfPosition(np.zeros(3))))
        self.assertTrue(np.all(np.zeros(3) == gridGeometry.indicesOfPosition(np.ones(3))))
        self.assertTrue(np.all(np.ones(3) == gridGeometry.indicesOfPosition(np.ones(3) + 0.1)))
        self.assertTrue(np.all(np.ones(3) == gridGeometry.indicesOfPosition(2*np.ones(3))))
        self.assertTrue(np.all(np.array([1,0,0]) == gridGeometry.indicesOfPosition(np.array([1.1, 1, 1]))))
        self.assertTrue(np.all(np.array([0,1,0]) == gridGeometry.indicesOfPosition(np.array([1, 1.1, 1]))))
        self.assertTrue(np.all(np.array([0,0,1]) == gridGeometry.indicesOfPosition(np.array([1, 1, 1.1]))))

    def test_positionFromIndices(self):
        # GIVEN 
        gridGeometry = UniformGridGeometry(8, np.zeros(3), 2*np.ones(3), False)

        # THEN
        self.assertTrue(np.allclose(np.zeros(3), gridGeometry.positionFromIndices(np.zeros(3).astype(int))))
        self.assertTrue(np.allclose(np.ones(3), gridGeometry.positionFromIndices(np.ones(3).astype(int))))
        self.assertTrue(np.allclose(np.array([1, 0, 0]), gridGeometry.positionFromIndices(np.array([1, 0, 0]))))
        self.assertTrue(np.allclose(np.array([0, 1, 0]), gridGeometry.positionFromIndices(np.array([0, 1, 0]))))
        self.assertTrue(np.allclose(np.array([0, 0, 1]), gridGeometry.positionFromIndices(np.array([0, 0, 1]))))

    def test_offsetFromIndices(self):
        # GIVEN
        gridGeometry = UniformGridGeometry(8, np.zeros(3), 2*np.ones(3), False)

        # THEN offsetFromIndices should map each position on the grid to a unique integer i such that 0 <= i <= 26
        npoints = gridGeometry.numPoints
        positions = itertools.product(range(npoints[0]), range(npoints[1]), range(npoints[2]))
        offsets = [gridGeometry.offsetFromIndices(np.array((x,y,z))) for (x,y,z) in positions]
        self.assertEquals(0, min(offsets))
        self.assertEquals(26, max(offsets))
        self.assertEquals(len(offsets), len(set(offsets)))

    def test_decimateSingleCellUnitDecimation(self):
        # GIVEN
        srcGrid = UniformGridGeometry(1, np.zeros(3), np.ones(3), False)
        decGrid = UniformGridGeometry()

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
        srcGrid = UniformGridGeometry(8, np.zeros(3), 2*np.ones(3), False)
        decGrid = UniformGridGeometry()

        # WHEN
        decGrid.decimate(srcGrid, 2)

        # THEN
        self.assertTrue(np.allclose(srcGrid.gridExtent, decGrid.gridExtent))
        self.assertTrue(np.allclose(srcGrid.minCorner, decGrid.minCorner))
        self.assertTrue(np.allclose(2*np.ones(3), decGrid.numPoints))
        self.assertTrue(np.allclose(2*np.ones(3), decGrid.cellExtent))
        self.assertTrue(np.allclose(0.5*np.ones(3), decGrid.cellsPerExtent))

    def test_copyShapeSingleCell(self):
        # GIVEN
        srcGrid = UniformGridGeometry(1, np.zeros(3), np.ones(3), False)
        decGrid = UniformGridGeometry()

        # WHEN
        decGrid.copyShape(srcGrid)

        # THEN
        self.assertTrue(np.allclose(srcGrid.gridExtent, decGrid.gridExtent))
        self.assertTrue(np.allclose(srcGrid.minCorner, decGrid.minCorner))
        self.assertTrue(np.allclose(srcGrid.numPoints , decGrid.numPoints))
        self.assertTrue(np.allclose(srcGrid.cellExtent, decGrid.cellExtent))
        self.assertTrue(np.allclose(srcGrid.cellsPerExtent, decGrid.cellsPerExtent))

if __name__ == '__main__':
    unittest.main()
