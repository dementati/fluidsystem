import propagator

class SimplePropagator(propagator.Propagator):
	
	def __init__(self, vectorField):
		self.vectorField = vectorField
