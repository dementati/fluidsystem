from __future__ import division

import numpy as np
from uniformgrid import UniformGrid

assignZOffsets = True


def computeJacobian(jacobian, vectorGrid):
    cellExtent = vectorGrid.cellExtent
    
    if cellExtent[2] > np.finfo(float).eps:
        reciprocalCellExtent = 1 / cellExtent
    else:
        reciprocalCellExtent = np.array([ 1 / cellExtent[0], 1 / cellExtent[1], 0 ])

    halfReciprocalCellExtent = reciprocalCellExtent / 2
    dims = np.copy(vectorGrid.numPoints)
    dimsMinus1 = dims - 1
    numXY = dims[0] * dims[1]
    
    def assignZOffsets():
        return (numXY * (index[2] - 1), 
                numXY * index[2], 
                numXY * (index[2] + 1))

    def assignYZOffsets():
        return (dims[0] * (index[1] - 1)    + offsetZ0,
                dims[0] * index[1]          + offsetZ0,
                dims[0] * (index[1] + 1)    + offsetZ0,
                dims[0] * index[1]          + offsetZM,
                dims[0] * index[1]          + offsetZP)

    def assignXYZOffsets():
        return (index[0]        + offsetY0Z0,
                index[0] - 1    + offsetY0Z0,
                index[0] + 1    + offsetY0Z0,
                index[0]        + offsetYMZ0,
                index[0]        + offsetYPZ0,
                index[0]        + offsetY0ZM,
                index[0]        + offsetY0ZP)

    index = np.zeros(3)
    for index[2] in range(dimsMinus1[2]):
        offsetZM, offsetZ0, offsetZP = assignZOffsets()

        for index[1] in range(dimsMinus1[1]):
            (offsetYMZ0, offsetY0Z0, offsetYPZ0, 
             offsetY0ZM, offsetY0ZP) = assignYZOffsets()

            for index[0] in range(dimsMinus1[0]):
                (offsetX0Y0Z0, offsetXMY0Z0, offsetXPY0Z0, 
                 offsetX0YMZ0, offsetX0YPZ0, offsetX0Y0ZM, 
                 offsetX0Y0ZP) = assignXYZOffsets()

                matrix = jacobian[offsetX0Y0Z0]
                matrix[0] = (vectorGrid[offsetXPY0Z0] - vectorGrid[offsetXMY0Z0]) * halfReciprocalCellExtent[0]
                matrix[1] = (vectorGrid[offsetX0YPZ0] - vectorGrid[offsetX0YMZ0]) * halfReciprocalCellExtent[1]
                matrix[2] = (vectorGrid[offsetX0Y0ZP] - vectorGrid[offsetX0Y0ZM]) * halfReciprocalCellExtent[2]

    def computeFiniteDiff():
        matrix = jacobian[offsetX0Y0Z0]

        if index[0] == 0:
            matrix[0] = matrix[0] # TODO: Finish this
