
import eyetracker
import cv2
import numpy as np
from movingaverage import MovingAveragePoints
from collections import namedtuple
from point import Point

class EyeCalibration:

	def __init__(self):
		self.camera = cv2.VideoCapture(0)
		self.tracker = eyetracker.EyeTracker()
		self.movAvgDict = dict()
		self.lookingPointMovAvg = MovingAveragePoints(
			Point(
				self.tracker.xScale / 2,
				self.tracker.yScale / 2
			), 5
		)
		self.subDict = dict()
		self.movAvgLength = 3

		self.topLeft = None
		self.topRight = None
		self.bottomLeft = None
		self.bottomRight = None
		self.center = None

		self.xBias = 0
		self.yBias = 0

		self.xMin = 0
		self.yMin = 0
		self.xMax = self.tracker.getXScale()
		self.yMax = self.tracker.getYScale()

		self.rangePadding = 20

		self.translationXMin = 0
		self.translationXMax = self.tracker.getXScale()
		self.translationYMin = 0
		self.translationYMax = self.tracker.getYScale()

	def getAverageLookingPoint(self, avgDict):
		avgX = 0
		avgY = 0
		sLen = len(avgDict.keys())

		for key in avgDict.keys():
			avgX += avgDict[key]["rVector"].getLastCompoundedResult().x / sLen
			avgY += avgDict[key]["rVector"].getLastCompoundedResult().y / sLen

		return Point(
			self.tracker.mapVal(
				self.tracker.getXScale() / 2 - avgX - self.xBias,
				self.xMin - self.rangePadding - self.xBias,
				self.xMax + self.rangePadding - self.xBias,
				self.translationXMin,
				self.translationXMax
			),
			self.tracker.mapVal(
				self.tracker.getYScale() / 2 + avgY - self.yBias,
				self.yMin - self.rangePadding - self.yBias,
				self.yMax + self.rangePadding - self.yBias,
				self.translationYMin,
				self.translationYMax
			)
		)

	def drawCanvas(self, img, avgDict):
		avgLookingPoint = self.getAverageLookingPoint(avgDict)
		currentPoint = self.lookingPointMovAvg.compound(
			avgLookingPoint,
			Point(0, 0)
		)
		cv2.circle(
			img, 
			currentPoint.toTuple(), 
			5, (0, 255, 255), 5
		)
		return self

	def updateExistingValue(self, r):
		r.pupil.centroid = self.movAvgDict[r.getId()]["centroid"].compound(
			r.getPupil().getCentroid(),
			Point(0, 0)
		)
		self.movAvgDict[r.getId()]["rVector"].compound(
			r.getResultantVector(
				self.tracker.getXScale(),
				self.tracker.getYScale()
			),
			self.movAvgDict[r.getId()][
				"centroid"
			].getLastCompoundedResult()
		)

	def updateMovAvgDict(self, results):
		newDict = dict()
		for r in results:

			if not r.getId() in self.movAvgDict.keys():
				self.movAvgDict[r.getId()] = {
					"centroid":
						MovingAveragePoints(
							r.getPupil().getCentroid(),
							self.movAvgLength
						),
					"rVector":
						MovingAveragePoints(
							r.getResultantVector(
								self.tracker.getXScale(),
								self.tracker.getYScale()
							),
							self.movAvgLength
						)
				}
			self.updateExistingValue(r)
			newDict[r.getId()] = self.movAvgDict[r.getId()]

		self.subDict = newDict
		return newDict

	def drawFilteredGaze(self, img, results):
		for r in results:
			hr = r.getHaarRectangle()
			imgDisp = cv2.resize(
				img[
					hr.y : hr.y + hr.h,
					hr.x : hr.x + hr.w
				],
				(
					self.tracker.getXScale(), 
					self.tracker.getYScale()
				)
			)

			centroidPoint = self.movAvgDict[
				r.getId()
			]["centroid"].getLastCompoundedResult()

			lookingVector = self.movAvgDict[
				r.getId()
			]["rVector"].getLastCompoundedResult()

			cv2.line(
				imgDisp,
				centroidPoint.toTuple(),
				(
					centroidPoint + 
					lookingVector
				).toTuple(),
				(0, 255, 0),
				20
			)

			img[
				hr.y : hr.y + hr.h, 
				hr.x : hr.x + hr.w
			] = cv2.resize(imgDisp, (hr.w, hr.h))

		return img

	def setPointAfterButton(self, button = 32):
		while True:
			_, frame = self.camera.read()
			self.tracker.setImage(frame)

			results = self.tracker.track()
			avgDict = self.updateMovAvgDict(results)
			avgPoint = self.getAverageLookingPoint(avgDict)

			raw_img = results.getImage()
			img = self.drawFilteredGaze(raw_img, results)

			flippedImage = cv2.flip(img, 1)
			self.drawCanvas(flippedImage, avgDict)
			cv2.imshow('Eye Tracking', flippedImage)
			if cv2.waitKey(1) == button:
				return avgPoint, results

	def getCenter(self, rect):
		try:
			return Point(
				rect.x + rect.w / 2, 
				rect.y + rect.h / 2
			)
		except AttributeError:
			return Point(0, 0)

	def setCornerPointsInteractive(self):
		self.lookingPointMovAvg.setLength(6)
		print "Set top left corner"
		self.topLeft, _ = self.setPointAfterButton()

		print "Set top right corner"
		self.topRight, _ = self.setPointAfterButton()

		print "Set bottom left corner"
		self.bottomLeft, _ = self.setPointAfterButton()

		print "Set bottom right corner"
		self.bottomRight, _ = self.setPointAfterButton()

		print "Set center"
		self.center, res = self.setPointAfterButton()

		self.xMax = self.topRight.x
		self.xMin = self.topLeft.x
		self.yMax = self.bottomRight.y
		self.yMin = self.topRight.y

	def getAverageBias(self):
		return (
			int(self.center.x - self.tracker.getXScale() / 2),
			int(self.center.y - self.tracker.getYScale() / 2)
		)

	def calibrate(self):
		self.setCornerPointsInteractive()
		self.xBias, self.yBias = self.getAverageBias()
		self.run()

	def run(self):
		self.lookingPointMovAvg.setLength(10)
		self.setPointAfterButton(27)

if __name__ == "__main__" or True:
	eyeCalibration = EyeCalibration()
	eyeCalibration.calibrate()

