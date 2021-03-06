from __future__ import division
import unittest
from ...src.vortex import uniformgrid as ug
from ...src.vortex import uniformgridmath as ugm
import logging
import numpy as np
import itertools

logger = logging.getLogger(__name__)

class TestUniformGridMath(unittest.TestCase):

    def test_computeJacobianZeroVectorGrid(self):
        # GIVEN
        vectorGrid = ug.UniformGrid(1, np.zeros(3), np.ones(3), False)
        vectorGrid.initialize(np.zeros(3))
        jacobian = ug.UniformGrid()
        jacobian.copyShape(vectorGrid)
        jacobian.initialize(np.zeros((3,3)))

        # WHEN
        ugm.computeJacobian(jacobian, vectorGrid)

        # THEN
        for coord in itertools.product(range(jacobian.numPoints[0]), 
                                       range(jacobian.numPoints[1]), 
                                       range(jacobian.numPoints[2])):
            npCoord = np.array(coord)
            self.assertTrue(np.allclose(np.zeros((3,3)), jacobian[jacobian.offsetFromIndices(npCoord)]))

    def test_computeJacobian2x2x2(self):
        # GIVEN
        vectorGrid = ug.UniformGrid(1, np.zeros(3), np.ones(3), False)
        vectorGrid.initialize(np.zeros(3))
        vectorGrid.insert(np.ones(3), np.ones(3))

        jacobian = ug.UniformGrid()
        jacobian.copyShape(vectorGrid)
        jacobian.initialize(np.zeros((3,3)))

        # WHEN
        ugm.computeJacobian(jacobian, vectorGrid)

        # THEN
        value = lambda npCoord: jacobian[jacobian.offsetFromIndices(npCoord)]
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([0,0,0]))))
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([1,0,0]))))
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([0,1,0]))))
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([0,0,1]))))

        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,1,0]))[0]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,1,0]))[1]))
        self.assertTrue(np.allclose(np.ones(3),      value(np.array([1,1,0]))[2]))

        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,0,1]))[0]))
        self.assertTrue(np.allclose(np.ones(3),      value(np.array([1,0,1]))[1]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,0,1]))[2]))

        self.assertTrue(np.allclose(np.ones(3),      value(np.array([0,1,1]))[0]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([0,1,1]))[1]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([0,1,1]))[2]))

        self.assertTrue(np.allclose(np.ones((3,3)),  value(np.array([1,1,1]))))

    def test_computeJacobian3x3x3(self):
        # GIVEN
        vectorGrid = ug.UniformGrid(8, np.zeros(3), 2*np.ones(3), False)
        vectorGrid.initialize(np.zeros(3))
        vectorGrid.insert(np.ones(3), np.ones(3))
        vectorGrid.insert(np.array([2, 1, 1]), 0.5*np.ones(3))

        jacobian = ug.UniformGrid()
        jacobian.copyShape(vectorGrid)
        jacobian.initialize(np.zeros((3,3)))

        # WHEN
        ugm.computeJacobian(jacobian, vectorGrid)

        # THEN
        value = lambda npCoord: jacobian[jacobian.offsetFromIndices(npCoord)]
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([0,0,0]))))
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([1,0,0]))))
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([0,1,0]))))
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([0,0,1]))))

        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,1,0]))[0]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,1,0]))[1]))
        self.assertTrue(np.allclose(np.ones(3),      value(np.array([1,1,0]))[2]))

        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,0,1]))[0]))
        self.assertTrue(np.allclose(np.ones(3),      value(np.array([1,0,1]))[1]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,0,1]))[2]))

        self.assertTrue(np.allclose(np.ones(3),      value(np.array([0,1,1]))[0]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([0,1,1]))[1]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([0,1,1]))[2]))

        self.assertTrue(np.allclose(0.25*np.ones(3), value(np.array([1,1,1]))[0]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,1,1]))[1]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,1,1]))[2]))
 
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([2,2,2]))))
        self.assertTrue(np.allclose(np.zeros((3,3)), value(np.array([1,2,2]))))

        self.assertTrue(np.allclose(np.zeros((3,3)),     value(np.array([2,1,2]))[0]))
        self.assertTrue(np.allclose(np.zeros((3,3)),     value(np.array([2,1,2]))[1]))
        self.assertTrue(np.allclose(-0.5*np.ones((3,3)), value(np.array([2,1,2]))[2]))
       
        self.assertTrue(np.allclose(np.zeros((3,3)),     value(np.array([2,2,1]))[0]))
        self.assertTrue(np.allclose(-0.5*np.ones((3,3)), value(np.array([2,2,1]))[1]))
        self.assertTrue(np.allclose(np.zeros((3,3)),     value(np.array([2,2,1]))[2]))

        self.assertTrue(np.allclose(-0.5*np.ones(3),  value(np.array([2,1,1]))[0]))
        self.assertTrue(np.allclose(np.zeros(3),      value(np.array([2,1,1]))[1]))
        self.assertTrue(np.allclose(np.zeros(3),      value(np.array([2,1,1]))[2]))

        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,1,2]))[0]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,1,2]))[1]))
        self.assertTrue(np.allclose(-np.ones(3),     value(np.array([1,1,2]))[2]))
        
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,2,1]))[0]))
        self.assertTrue(np.allclose(-np.ones(3),     value(np.array([1,2,1]))[1]))
        self.assertTrue(np.allclose(np.zeros(3),     value(np.array([1,2,1]))[2]))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
