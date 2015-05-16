from __future__ import division
import numpy as np
import sys
from .. import mathutil as mu
from .. import assertutil as au
import logging

logger = logging.getLogger(__name__)

class UniformGridGeometry(object):

    nudge = 1 + np.finfo(float).eps

    def __init__(self, numElements = None, vMin = None, vMax = None, powerOf2 = None):
        if numElements is None and vMin is None and vMax is None:
            self.clear()
        else:
            assert isinstance(numElements, int), "numElements must be an integer"
            assert numElements > 0, "numElements must be non-negative"
            au.assertVec2or3(vMin, "vMin")
            au.assertVec2or3(vMax, "vMax")
            assert len(vMin) == len(vMax), "vMin and vMax must be of the same dimensions"
            assert isinstance(powerOf2, bool), "powerOf2 must be a boolean"

            if len(vMin) == 2:
                vMin = np.array([vMin[0], vMin[1], 0])
                vMax = np.array([vMax[0], vMax[1], 0])

            self.defineShape(numElements, vMin, vMax, powerOf2)

    
    def defineShape(self, numElements, vMin, vMax, powerOf2):
        assert isinstance(numElements, int), "numElements must be an integer"
        assert numElements > 0, "numElements must be non-negative"
        au.assertVec3(vMin, "vMin")
        au.assertVec3(vMax, "vMax")
        assert isinstance(powerOf2, bool), "powerOf2 must be a boolean"

        logger.debug("defineShape called with:")
        logger.debug("numElements = %d" % numElements)
        logger.debug("vMin = %s" % vMin)
        logger.debug("vMax = %s" % vMax)
        logger.debug("powerOf2 = %s" % str(powerOf2))

        self.setMinCorner(vMin)
        self.setGridExtent((vMax - vMin) * UniformGridGeometry.nudge)

        sizeEffective = np.copy(self.gridExtent)
        numDims = 3
        for i in range(3):
            if 0 == sizeEffective[i]:
                sizeEffective[i] = 1
                self.gridExtent[i] = 0
                numDims -= 1
        logger.debug("sizeEffective = %s" % sizeEffective)

        volume = np.prod(sizeEffective)
        logger.debug("volume = %f" % volume)

        cellVolumeCubeRoot = np.power( volume/numElements, -1/numDims )
        logger.debug("cellVolumeCubeRoot = %f" % cellVolumeCubeRoot)

        numCells = np.maximum(np.ones(3), (self.gridExtent * cellVolumeCubeRoot + 0.5).astype(int)).astype(int)
        logger.debug("self.gridExtent * cellVolumeCubeRoot + 0.5 = %s", self.gridExtent * cellVolumeCubeRoot + 0.5)
        logger.debug("numCells = %s" % numCells)

        if powerOf2:
            numCells = mu.nearestPowerOf2(numCells)
            logger.debug("rounded to nearest power of 2, numCells = %s" % numCells)

        while np.prod(numCells) >= numElements * 8:
            numCells = np.maximum(np.ones(3), numCells/2).astype(int)
            logger.debug("np.prod(numCells) >= numElements * 8, numCells = %s" % numCells)

        self.setNumPoints(numCells + 1)

        self.precomputeSpacing()

    def decimate(self, src, decimation):
        assert isinstance(src, UniformGridGeometry), "src must be a UniformGridGeometry"
        assert isinstance(decimation, int), "decimation must be an integer"
        assert decimation > 0, "decimation must be positive"
        assert decimation == 1 or np.all(src.getNumCells() > 1), "cannot decimate a single-cell axis"

        self.setGridExtent(src.gridExtent)
        self.setMinCorner(src.minCorner)
        self.setNumPoints((src.getNumCells() / decimation).astype(int) + 1)

        if decimation > 1:
            self.setNumPoints(np.maximum((2 * np.ones(3)).astype(int), self.numPoints))

        self.precomputeSpacing()

    def copyShape(self, src):
        assert isinstance(src, UniformGridGeometry), "src must be a UniformGridGeometry"

        self.decimate(src, 1)

    def precomputeSpacing(self):
        self.setCellExtent(self.gridExtent / self.getNumCells())

        cellsPerExtent = np.zeros(3)
        if 0 == self.gridExtent[2]:
            cellsPerExtent[:2] = self.getNumCells()[:2] / self.gridExtent[:2]
            cellsPerExtent[2] = 1/np.finfo(float).tiny
        else:
            cellsPerExtent = self.getNumCells() / self.gridExtent

        self.setCellsPerExtent(cellsPerExtent)

    def setNumPoints(self, numPoints):
        au.assertIntVec3(numPoints, "numPoints")
        self.numPoints = numPoints
        logger.debug("set numPoints to %s" % self.numPoints)

    def getNumCells(self):
        return self.numPoints - 1

    def setCellsPerExtent(self, cellsPerExtent):
        au.assertVec3(cellsPerExtent, "cellsPerExtent")
        self.cellsPerExtent = cellsPerExtent
        logger.debug("set cellsPerExtent to %s" % self.cellsPerExtent)

    def setMinCorner(self, minCorner):
        au.assertVec3(minCorner, "minCorner")
        self.minCorner = minCorner
        logger.debug("set minCorner to %s" % self.minCorner)

    def setGridExtent(self, gridExtent):
        au.assertVec3(gridExtent, "gridExtent")
        self.gridExtent = gridExtent
        logger.debug("set gridExtent to %s" % self.gridExtent)

    def setCellExtent(self, cellExtent):
        au.assertVec3(cellExtent, "cellExtent")
        self.cellExtent = cellExtent
        logger.debug("set cellExtent to %s" % self.cellExtent)

    def indicesOfPosition(self, position):
        au.assertVec3(position, "position")
        posRel = position - self.minCorner
        return (posRel * self.cellsPerExtent).astype(int)

    def positionFromIndices(self, indices):
        au.assertIntVec3(indices, "indices")
        return self.minCorner + indices * self.cellExtent

    def offsetFromIndices(self, indices):
        au.assertIntVec3(indices, "indices")
        assert np.all(indices >= 0), "All indices must be non-negative"
        assert np.all(indices < self.numPoints), "Indices out of bounds"

        offset = indices[0] + self.numPoints[0] * ( indices[1] + self.numPoints[1] * indices[2] )
        assert issubclass(type(offset), np.integer), "Grid is in inconsistent state, numPoints are not integers"

        return int(offset)

    def getGridCapacity(self):
        return np.prod(self.numPoints)

    def clear(self):
        self.setMinCorner(np.zeros(3))
        self.setGridExtent(np.zeros(3))
        self.setCellExtent(np.zeros(3))
        self.setCellsPerExtent(np.zeros(3))
        self.setNumPoints(np.zeros(3).astype(int))
