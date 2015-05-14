from __future__ import division

import numpy as np
import sys
from .. import mathutil as mu

class UniformGridGeometry:

    nudge = 1 + sys.float_info.epsilon

    def __init__(self, numElements = None, vMin = None, vMax = None, powerOf2 = None):
        if numElements is None and vMin is None and vMax is None:
			self.clear()
        else:
            self.defineShape(numElements, vMin, vMax, powerOf2)

    
    def defineShape(self, numElements, vMin, vMax, powerOf2):
        self.minCorner = vMin
        self.gridExtent = (vMax - vMin) * nudge
        
        sizeEffective = self.gridExtent
        numDims = 3
        for i in range(3):
            if 0 == sizeEffective[i]:
                sizeEffective[i] = 1
                self.gridExtent[i] = 0
                numDims -= 1

        volume = np.prod(sizeEffective)
        cellVolumeCubeRoot = np.power( volume/numElements, -1/numDims )

        numCells = np.maximum(np.ones(3), self.gridExtent * cellVolumeCubeRoot + 0.5)
        
        if powerOf2:
            numCells = mu.nearestPowerOf2(numCells)

        while np.prod(numCells) >= numElements * 8:
            numCells = np.maximum(np.ones(3), numCells/2)

        self.numPoints = numCells + 1

        self.precomputeSpacing()

	def decimate(self, src, decimation):
		self.gridExtent = src.gridExtent
		self.minCorner = src.minCorner
		self.numPoints = src.getNumCells() / decimation + 1

		if decimation > 1:
			self.numPoints = np.maximum(2 * np.ones(3), self.numPoints)

		self.precomputeSpacing()

	def copyShape(self, src):
		self.decimate(src, 1)

    def precomputeSpacing(self):
        self.cellExtent = self.gridExtent / self.getNumCells()
        self.cellsPerExtent = self.getNumCells() / self.gridExtent

        if 0 == self.gridExtent[2]:
            self.cellsPerExtent[2] = 1/sys.float_info.min

    def getNumCells(self):
        return self.numPoints - 1

    def indicesOfPosition(self, position):
        posRel = position - self.minCorner
        return (posRel * self.cellsPerExtent).astype(int)

    def positionFromIndices(self, indices):
        return self.minCorner + indices * self.cellExtent

    def offsetFromIndices(self, indices):
        return indices[0] + self.numPoints[0] * ( indices[1] + self.numPoints[1] * indices[2] )

    def getGridCapacity(self):
        return np.prod(self.numPoints)

	def clear(self):
		self.minCorner = np.zeros(3)
		self.gridExtent = np.zeros(3)
		self.cellExtent  = np.zeros(3)
		self.cellsPerExtent = np.zeros(3)
		self.numPoints = np.zeros(3)
