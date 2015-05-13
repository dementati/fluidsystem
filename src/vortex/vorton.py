from __future__ import division

import numpy as np

class Vorton:

    def __init__(self, position=None, vorticity=None):
        self.position = np.array([0.0, 0.0, 0.0]) if position is None else position
        self.vorticity = np.array([0.0, 0.0, 0.0]) if vorticity is None else vorticity
