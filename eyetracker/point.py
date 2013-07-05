
class Point:

	def __init__(self, x, y):
		self.x = x
		self.y = y

	def toTuple(self):
		return (self.x, self.y)

	def __add__(self, p):
		return Point(p[0] + self[0], p[1] + self[1])

	def __sub__(self, p):
		return Point(self[0] - p[0], self[1] - p[1])

	def __getitem__(self, key):
		if key == 0:
			return self.x

		elif key == 1:
			return self.y

		else:
			raise IndexError("YOU FOOL")
