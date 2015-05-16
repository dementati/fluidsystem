from __future__ import division
import numpy as np
from uniformgridgeometry import UniformGridGeometry
from .. import assertutil as au

class UniformGrid(UniformGridGeometry):

    def __init__(self, numElements=None, vMin=None, vMax=None, powerOf2=None):
        if numElements != None or vMin != None or vMax != None or powerOf2 != None:
            assert isinstance(numElements, int), "numElements must be an integer"
            au.assertVec3(vMin, "vMin")
            au.assertVec3(vMax, "vMax")
            assert isinstance(powerOf2, bool), "powerOf2 must be a boolean"

        super(UniformGrid, self).__init__(numElements, vMin, vMax, powerOf2)

    def initialize(self, defaultValue=None):
        self.contents = [defaultValue] * self.getGridCapacity()

    def defineShape(self, numElements, vMin, vMax, powerOf2):
        assert isinstance(numElements, int), "numElements must be an integer"
        au.assertVec3(vMin, "vMin")
        au.assertVec3(vMax, "vMax")
        assert isinstance(powerOf2, bool), "powerOf2 must be a boolean"

        self.contents = []
        super(UniformGrid, self).defineShape(numElements, vMin, vMax, powerOf2)

    def decimate(self, src, decimation):
        assert isinstance(src, UniformGrid), "src must be a UniformGrid"
        assert isinstance(decimation, int), "decimation must be an integer"

        super(UniformGrid, self).decimate(src, decimation)

    def interpolate(self, position):
        au.assertVec3(position, "position")
        self.assertInitialized()

        indices = self.indicesOfPosition(position)
        minCorner = self.positionFromIndices(indices)

        offsetX0Y0Z0 = self.offsetFromIndices(indices)
        diff = position - minCorner
        tween = diff * self.cellsPerExtent
        oneMinusTween = np.ones(3) - tween
        numXY = np.prod(self.numPoints[:2])
        offsetX1Y0Z0  = offsetX0Y0Z0 + 1 
        offsetX0Y1Z0  = offsetX0Y0Z0 + self.numPoints[0] 
        offsetX1Y1Z0  = offsetX0Y0Z0 + self.numPoints[0] + 1 
        offsetX0Y0Z1  = offsetX0Y0Z0 + numXY 
        offsetX1Y0Z1  = offsetX0Y0Z0 + numXY + 1 
        offsetX0Y1Z1  = offsetX0Y0Z0 + numXY + self.numPoints[0] 
        offsetX1Y1Z1  = offsetX0Y0Z0 + numXY + self.numPoints[0] + 1
        
        return  ( oneMinusTween[0] * oneMinusTween[1] * oneMinusTween[2] * self.contents[ offsetX0Y0Z0 ]
                +         tween[0] * oneMinusTween[1] * oneMinusTween[2] * self.contents[ offsetX1Y0Z0 ]
                + oneMinusTween[0] *         tween[1] * oneMinusTween[2] * self.contents[ offsetX0Y1Z0 ]
                +         tween[0] *         tween[1] * oneMinusTween[2] * self.contents[ offsetX1Y1Z0 ]
                + oneMinusTween[0] * oneMinusTween[1] *         tween[2] * self.contents[ offsetX0Y0Z1 ]
                +         tween[0] * oneMinusTween[1] *         tween[2] * self.contents[ offsetX1Y0Z1 ]
                + oneMinusTween[0] *         tween[1] *         tween[2] * self.contents[ offsetX0Y1Z1 ]
                +         tween[0] *         tween[1] *         tween[2] * self.contents[ offsetX1Y1Z1 ] )


    def insert(self, position, item):
        au.assertVec3(position, "position")
        self.assertInitialized()

        indices = self.indicesOfPosition(position)
        minCorner = self.positionFromIndices(indices)

        offsetX0Y0Z0 = self.offsetFromIndices(indices)
        diff = position - minCorner
        tween = diff * self.cellsPerExtent
        oneMinusTween = np.ones(3) - tween
        numXY = np.prod(self.numPoints[:2])
        offsetX1Y0Z0  = offsetX0Y0Z0 + 1 
        offsetX0Y1Z0  = offsetX0Y0Z0 + self.numPoints[0] 
        offsetX1Y1Z0  = offsetX0Y0Z0 + self.numPoints[0] + 1 
        offsetX0Y0Z1  = offsetX0Y0Z0 + numXY 
        offsetX1Y0Z1  = offsetX0Y0Z0 + numXY + 1 
        offsetX0Y1Z1  = offsetX0Y0Z0 + numXY + self.numPoints[0] 
        offsetX1Y1Z1  = offsetX0Y0Z0 + numXY + self.numPoints[0] + 1

        self.contents[ offsetX0Y0Z0 ] += oneMinusTween[0] * oneMinusTween[1] * oneMinusTween[2] * item ;
        self.contents[ offsetX1Y0Z0 ] +=         tween[0] * oneMinusTween[1] * oneMinusTween[2] * item ;
        self.contents[ offsetX0Y1Z0 ] += oneMinusTween[0] *         tween[1] * oneMinusTween[2] * item ;
        self.contents[ offsetX1Y1Z0 ] +=         tween[0] *         tween[1] * oneMinusTween[2] * item ;
        self.contents[ offsetX0Y0Z1 ] += oneMinusTween[0] * oneMinusTween[1] *         tween[2] * item ;
        self.contents[ offsetX1Y0Z1 ] +=         tween[0] * oneMinusTween[1] *         tween[2] * item ;
        self.contents[ offsetX0Y1Z1 ] += oneMinusTween[0] *         tween[1] *         tween[2] * item ;
        self.contents[ offsetX1Y1Z1 ] +=         tween[0] *         tween[1] *         tween[2] * item ;

    def clear(self):
        self.contents = []
        super(UniformGrid, self).clear()

    def assertInitialized(self):
        assert bool(self.contents), "Grid has not been initialized"
