
from collections import namedtuple
import numpy as np
from functools import partial
from point import Point

class MovingAverageList(object):

	def __init__(self, startingValue, length):
		self.movAvgList = [startingValue for _ in xrange(length)]
		self.lastMean = None

	def getLength(self):
		return len(self.movAvgList)

	def put(self, value):
		del self.movAvgList[0]
		self.movAvgList += [value]
		return self

	def compound(self, value, filterFunc):
		self.put(value)
		maListCopy = list()
		maListCopy.extend(self.movAvgList)
		self.lastMean = self.getMean(
			filterFunc(maListCopy)
		)
		return self.lastMean

	def setLength(self, length):
		if length < len(self.movAvgList):
			self.movAvgList = self.movAvgList[length - 1 : -1]
		elif length > len(self.movAvgList):
			lPoint = self.movAvgList[-1]
			self.movAvgList.extend(
				[
					lPoint for _ in xrange(
						length - len(self.movAvgList)
					)
				]
			)
		return self

	def getLastCompoundedResult(self):
		return self.lastMean

	def __getitem__(self, key):
		return self.movAvgList[key]

	def __str__(self):
		return str(self.movAvgList)

class MovingAveragePoints(MovingAverageList):
	def __init__(self, startingValue, length):
		super(
			MovingAveragePoints, 
			self
		).__init__(startingValue, length)
		self.lastMean = Point(0, 0)

	def compound(self, value, refPoint):
		return super(
			MovingAveragePoints, 
			self
		).compound(value, partial(self.removeOutliers, refPoint))

	def removeOutliers(self, refPoint, maList, m = 3):
		distList = map(
			partial(self.norm, refPoint), 
			maList
		)
		meanDist = np.mean(distList)
		stdDist = np.std(distList)
		return [
			maList[i] for i, dist in enumerate(distList) 
			if dist < meanDist + m * stdDist
		]

	def getMean(self, maList):
		if len(maList) == 0:
			maList = self.movAvgList

		divList = map(
			lambda val: Point(
				float(val.x) / float(len(maList)), 
				float(val.y) / float(len(maList))
			), 
			maList
		)
		retVal = reduce(
			lambda b1, b2: Point(
				b1.x + b2.x, b1.y + b2.y
			), 
			divList
		)
		return Point(int(retVal.x), int(retVal.y))

	def norm(self, p1, p2):
		"""
		Finds the distance between the two points
		"""
		return np.sqrt(
			pow(p1[0] - p2[0], 2) + 
			pow(p1[1] - p2[1], 2)
		)

