from __future__ import division
import numpy as np
import uniformgrid as ug
import logging

logger = logging.getLogger(__name__)

class NestedGrid(object):
    
    def __init__(self, src=None):
        self.decimations = 0
        self.layers = []
    
        if src:
            self.initialize(src)

    def initialize(self, src):
        self.layers = []
        numLayers = self.precomputeNumLayers(src)
        self.layers = [None]*numLayers
        self.addLayer(src, 1)
        index = 1
        while self.layers[index - 1].getGridCapacity() > 8:
            self.addLayer(self.layers[index - 1], 2)
            index += 1

        self.precomputeDecimations()

    def precomputeDecimations(self):
        numLayers = len(self.layers)
        self.decimations = [None]*numLayers
        for layer in range(1, numLayers):
            self.computeDecimations(decimations[layer], layer)
        self.decimations[0] = np.zeros(3).astype(int)

    def computeDecimations(self, decimations, parentLayer):
        parent = self.layers[parentLayer]
        child = self.layers[parentLayer - 1]

        decimations[0] = (child.getNumCells(0) / parent.getNumCells(0)).astype(int)
        decimations[1] = (child.getNumCells(1) / parent.getNumCells(1)).astype(int)
        decimations[2] = (child.getNumCells(2) / parent.getNumCells(2)).astype(int)

    def addLayer(self, layerTemplate, decimation):
        layers.append(ug.UniformGrid())
        layers[-1].decimate(layerTemplate, decimation)
        layers.initialize()

    def precomputeNumLayers(self, src):
        numLayers = 1
        numPoints = np.copy(src.numPoints)
        size = np.prod(numPoints)
        while size > 8:
            numLayers += 1
            numPoints = np.maximum(np.ones(3), ((numPoints - 1) / 2).astype(int)).astype(int) + 1
            size = np.prod(numPoints)

        return numLayers
