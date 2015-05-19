from __future__ import division

import numpy as np
from uniformgrid import UniformGrid
import logging

logger = logging.getLogger(__name__)

def computeJacobian(jacobian, vectorGrid):
    cellExtent = vectorGrid.cellExtent
    logger.debug("cellExtent = %s" % cellExtent)

    if cellExtent[2] > np.finfo(float).eps:
        reciprocalCellExtent = 1 / cellExtent
    else:
        reciprocalCellExtent = np.array([ 1 / cellExtent[0], 1 / cellExtent[1], 0 ])
    logger.debug("reciprocalCellExtent = %s" % reciprocalCellExtent)

    halfReciprocalCellExtent = reciprocalCellExtent / 2
    logger.debug("halfReciprocalCellExtent = %s" % halfReciprocalCellExtent)
    dims = np.copy(vectorGrid.numPoints)
    logger.debug("dims = %s" % dims)
    dimsMinus1 = dims - 1
    logger.debug("dimsMinus1 = %s" % dimsMinus1)
    numXY = dims[0] * dims[1]
    logger.debug("numXY = %s" % numXY)
    
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

    def printOffsets():
        logger.debug("index = %s" % index)
        logger.debug("offsetXMY0Z0 = %s" % offsetXMY0Z0)
        logger.debug("offsetX0Y0Z0 = %s" % offsetX0Y0Z0)
        logger.debug("offsetXPY0Z0 = %s" % offsetXPY0Z0)

        logger.debug("offsetX0YMZ0 = %s" % offsetX0YMZ0)
        logger.debug("offsetX0Y0Z0 = %s" % offsetX0Y0Z0)
        logger.debug("offsetX0YPZ0 = %s" % offsetX0YPZ0)

        logger.debug("offsetX0Y0ZM = %s" % offsetX0Y0ZM)
        logger.debug("offsetX0Y0Z0 = %s" % offsetX0Y0Z0)
        logger.debug("offsetX0Y0ZP = %s" % offsetX0Y0ZP)


    index = np.zeros(3).astype(int)
    for index[2] in range(1, dimsMinus1[2]):
        offsetZM, offsetZ0, offsetZP = assignZOffsets()

        for index[1] in range(1, dimsMinus1[1]):
            (offsetYMZ0, offsetY0Z0, offsetYPZ0, 
             offsetY0ZM, offsetY0ZP) = assignYZOffsets()

            for index[0] in range(1, dimsMinus1[0]):
                (offsetX0Y0Z0, offsetXMY0Z0, offsetXPY0Z0, 
                 offsetX0YMZ0, offsetX0YPZ0, offsetX0Y0ZM, 
                 offsetX0Y0ZP) = assignXYZOffsets()

                printOffsets()
                
                matrix = jacobian[offsetX0Y0Z0]
                matrix[0] = (vectorGrid[offsetXPY0Z0] - vectorGrid[offsetXMY0Z0]) * halfReciprocalCellExtent[0]
                matrix[1] = (vectorGrid[offsetX0YPZ0] - vectorGrid[offsetX0YMZ0]) * halfReciprocalCellExtent[1]
                matrix[2] = (vectorGrid[offsetX0Y0ZP] - vectorGrid[offsetX0Y0ZM]) * halfReciprocalCellExtent[2]

    def computeFiniteDiff():
        matrix = jacobian[offsetX0Y0Z0]

        if index[0] == 0:                   matrix[0] = (vectorGrid[offsetXPY0Z0] - vectorGrid[offsetX0Y0Z0]) * reciprocalCellExtent[0]
        elif index[0] ==  dimsMinus1[0]:    matrix[0] = (vectorGrid[offsetX0Y0Z0] - vectorGrid[offsetXMY0Z0]) * reciprocalCellExtent[0]
        else:                               matrix[0] = (vectorGrid[offsetXPY0Z0] - vectorGrid[offsetXMY0Z0]) * halfReciprocalCellExtent[0]

        if index[1] == 0:                   matrix[1] = (vectorGrid[offsetX0YPZ0] - vectorGrid[offsetX0Y0Z0]) * reciprocalCellExtent[1]
        elif index[1] ==  dimsMinus1[1]:    matrix[1] = (vectorGrid[offsetX0Y0Z0] - vectorGrid[offsetX0YMZ0]) * reciprocalCellExtent[1]
        else:                               matrix[1] = (vectorGrid[offsetX0YPZ0] - vectorGrid[offsetX0YMZ0]) * halfReciprocalCellExtent[1]

        if index[2] == 0:                   matrix[2] = (vectorGrid[offsetX0Y0ZP] - vectorGrid[offsetX0Y0Z0]) * reciprocalCellExtent[2]
        elif index[2] ==  dimsMinus1[2]:    matrix[2] = (vectorGrid[offsetX0Y0Z0] - vectorGrid[offsetX0Y0ZM]) * reciprocalCellExtent[2]
        else:                               matrix[2] = (vectorGrid[offsetX0Y0ZP] - vectorGrid[offsetX0Y0ZM]) * halfReciprocalCellExtent[2]

    index[0] = 0
    for index[2] in range(dims[2]):
        offsetZM, offsetZ0, offsetZP = assignZOffsets()
        
        for index[1] in range(dims[1]):
            (offsetYMZ0, offsetY0Z0, offsetYPZ0, 
             offsetY0ZM, offsetY0ZP) = assignYZOffsets()

            (offsetX0Y0Z0, offsetXMY0Z0, offsetXPY0Z0, 
             offsetX0YMZ0, offsetX0YPZ0, offsetX0Y0ZM, 
             offsetX0Y0ZP) = assignXYZOffsets()

            computeFiniteDiff()
   
    index[1] = 0
    for index[2] in range(dims[2]):
        offsetZM, offsetZ0, offsetZP = assignZOffsets()

        (offsetYMZ0, offsetY0Z0, offsetYPZ0, 
         offsetY0ZM, offsetY0ZP) = assignYZOffsets()

        for index[0] in range(dims[0]):
            (offsetX0Y0Z0, offsetXMY0Z0, offsetXPY0Z0, 
             offsetX0YMZ0, offsetX0YPZ0, offsetX0Y0ZM, 
             offsetX0Y0ZP) = assignXYZOffsets()

            computeFiniteDiff()

    index[2] = 0
    offsetZM, offsetZ0, offsetZP = assignZOffsets()
    
    for index[1] in range(dims[1]):
        (offsetYMZ0, offsetY0Z0, offsetYPZ0, 
         offsetY0ZM, offsetY0ZP) = assignYZOffsets()

        for index[0] in range(dims[0]):
            (offsetX0Y0Z0, offsetXMY0Z0, offsetXPY0Z0, 
             offsetX0YMZ0, offsetX0YPZ0, offsetX0Y0ZM, 
             offsetX0Y0ZP) = assignXYZOffsets()

            computeFiniteDiff()

    index[0] = dimsMinus1[0]
    for index[2] in range(dims[2]):
        offsetZM, offsetZ0, offsetZP = assignZOffsets()
        
        for index[1] in range(dims[1]):
            (offsetYMZ0, offsetY0Z0, offsetYPZ0, 
             offsetY0ZM, offsetY0ZP) = assignYZOffsets()

            (offsetX0Y0Z0, offsetXMY0Z0, offsetXPY0Z0, 
             offsetX0YMZ0, offsetX0YPZ0, offsetX0Y0ZM, 
             offsetX0Y0ZP) = assignXYZOffsets()

            computeFiniteDiff()
   
    index[1] = dimsMinus1[1]
    for index[2] in range(dims[2]):
        offsetZM, offsetZ0, offsetZP = assignZOffsets()

        (offsetYMZ0, offsetY0Z0, offsetYPZ0, 
         offsetY0ZM, offsetY0ZP) = assignYZOffsets()

        for index[0] in range(dims[0]):
            (offsetX0Y0Z0, offsetXMY0Z0, offsetXPY0Z0, 
             offsetX0YMZ0, offsetX0YPZ0, offsetX0Y0ZM, 
             offsetX0Y0ZP) = assignXYZOffsets()

            computeFiniteDiff()

    index[2] = dimsMinus1[2]
    offsetZM, offsetZ0, offsetZP = assignZOffsets()
    
    for index[1] in range(dims[1]):
        (offsetYMZ0, offsetY0Z0, offsetYPZ0, 
         offsetY0ZM, offsetY0ZP) = assignYZOffsets()

        for index[0] in range(dims[0]):
            (offsetX0Y0Z0, offsetXMY0Z0, offsetXPY0Z0, 
             offsetX0YMZ0, offsetX0YPZ0, offsetX0Y0ZM, 
             offsetX0Y0ZP) = assignXYZOffsets()

            computeFiniteDiff()

