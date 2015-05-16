import numpy as np

def assertVec2or3(x, name):
    assert isinstance(x, np.ndarray), "%s must be a numpy array" % name
    assert np.ndim(x) == 1, "%s must be a vector" % name
    assert len(x) == 2 or len(x) == 3, "%s must be a 2-vector or a 3-vector" % name

def assertVec3(x, name):
    assert isinstance(x, np.ndarray), "%s must be a numpy array" % name
    assert np.ndim(x) == 1, "%s must be a vector" % name
    assert len(x) == 3, "%s must be a 3-vector" % name

def assertIntVec3(x, name):
    assertVec3(x, name)
    assert issubclass(x.dtype.type, np.integer), "%s must be integers" % name
