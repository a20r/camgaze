
import eyetracker
import cv2
import numpy as np
from movingaverage import MovingAveragePoints
from collections import namedtuple
from point import Point

"""
TODO:
	- Train adaptive threshold during calibration phase :- CHECK
	- CLEAN THIS SHIT UP 
	- Create a function relating angle position from 
	  corners to size of the blob to make sure that the
	  the correct eye is being found
	- Port the algorithm to JavaScript
	- Use a t-test to specify when there is a change in 
	  position using the moving average. So after filtering
	  the outliers out, perform a t-test and if there
	  is a statistical difference, update the position.
"""

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

		self.movAvgLength = 3

		self.bottomRight = None

		self.xMin = 0
		self.yMin = 0
		self.xMax = self.tracker.getXScale()
		self.yMax = self.tracker.getYScale()

		self.rangePadding = 10

		self.headCenter = None

		self.translationXMin = 0
		self.translationYMin = 0
		(
			self.translationXMax, 
			self.translationYMax
		) = (
			self.tracker.getXScale(), 
			self.tracker.getYScale()
		)

		self.canvasDim = Point(1280, 770)

		self.colorPadding = 5

		self.learningColors = dict()

	def getAverageLookingPoint(self, avgDict):
		avgX = 0
		avgY = 0
		sLen = len(avgDict.keys())

		for key in avgDict.keys():
			avgX += avgDict[key]["rVector"].getLastCompoundedResult().x / sLen
			avgY += avgDict[key]["rVector"].getLastCompoundedResult().y / sLen
		mirroredAvgX = self.tracker.getXScale() / 2 - avgX
		mirroredAvgY = self.tracker.getYScale() / 2 + avgY

		return Point(
			self.tracker.mapVal(
				mirroredAvgX,
				self.xMin - self.rangePadding,
				self.xMax + self.rangePadding,
				self.translationXMin,
				self.translationXMax
			),
			self.tracker.mapVal(
				mirroredAvgY,
				self.yMin - self.rangePadding,
				self.yMax + self.rangePadding,
				self.translationYMin,
				self.translationYMax
			)
		)

	def getBoundaryRect(self, point, rects):
		for r in rects:
			if point.x <= r.x or point.x >= r.x + r.w:
				continue
			if point.y <= r.y + r.h and point.y >= r.y:
				return r

	def drawAllRectangles(self, img, rects):
		for r in rects:
			cv2.rectangle(
				img,
				(
					r.x,
					r.y
				),
				(
					r.x + r.w,
					r.y + r.h
				),
				(0, 0, 0),
				5
			)


	def drawCanvas(self, img, avgDict, res):
		avgLookingPoint = self.getAverageLookingPoint(avgDict)

		# uses (0, 0) as reference point because of scaling
		currentPoint = self.lookingPointMovAvg.compound(
			avgLookingPoint,
			Point(0, 0)
		)

		if self.headCenter != None:
			currentPoint -= (
				self.getAverageEyePosition(res) - 
				self.headCenter
			)

		canvasPoint = Point(
			self.tracker.mapVal(
				currentPoint.x,
				self.translationXMin,
				self.translationXMax,
				0,
				self.canvasDim.x
			),
			self.tracker.mapVal(
				currentPoint.y,
				self.translationYMin,
				self.translationYMax,
				0,
				self.canvasDim.y
			)
		)

		try:
			raise AttributeError
			boundaryRect = self.getBoundaryRect(
				canvasPoint, 
				self.rects
			)
			cv2.rectangle(
				img,
				(
					boundaryRect.x,
					boundaryRect.y
				),
				(
					boundaryRect.x + boundaryRect.w,
					boundaryRect.y + boundaryRect.h
				),
				(255, 0, 0),
				-1
			)
		except AttributeError:
			pass

		#self.drawAllRectangles(img, self.rects)
		
		cv2.circle(
			img, 
			canvasPoint.toTuple(), 
			5, (0, 0, 255), 5
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

	# updates the moving average dictionary
	# returns a dictionary of the current eyes
	# dictionary structure :-
	# eye_id : {
	#		"centroid" : MovingAverage(eye_position) 
	#						--> length = self.movAvgLength
	#		"rVector" : MovingAverage(resultant_looking_vector) 
	#						--> length = self.movAvgLength
	# }
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

	def updateLearningColors(self, results):
		for r in results:
			try:
				self.learningColors[r.getId()] += [r.getMaxMinColors()]
			except KeyError:
				self.learningColors[r.getId()] = [r.getMaxMinColors()]

	# main function that gets the frame
	# gets statistics about the frame 
	# after a specified button is pressed
	def setPointAfterButton(self, button = 32, circlePosition = None, isLearning = True):
		while True:
			_, frame = self.camera.read()
			self.tracker.setImage(frame)

			results = self.tracker.track()
			avgDict = self.updateMovAvgDict(results)
			avgPoint = self.getAverageLookingPoint(avgDict)
			raw_img = results.getImage()

			canvas = 255 * np.ones_like(raw_img)
			canvas = cv2.resize(canvas, self.canvasDim.toTuple())

			img = self.drawFilteredGaze(raw_img, results)

			img = cv2.resize(img, (1280, 770))

			flippedImage = cv2.flip(img, 1)

			canvas[0:770, 0:1280] = flippedImage

			self.drawCanvas(canvas, avgDict, results)

			if circlePosition != None:
				cv2.circle(
					canvas, 
					circlePosition, 
					20, (255, 0, 255), -1
				)

			if isLearning:
				self.updateLearningColors(results)

			cv2.imshow('Eye Tracking', canvas)
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

	# points used for calibration
	def generateCalibrationPoints(self, rowNum, _colNum = None):
		colNum = rowNum if _colNum == None else _colNum
		xInc = self.canvasDim.x / (rowNum - 1)
		yInc = self.canvasDim.y / (colNum - 1)
		return [
			(
				i * xInc, 
				j * yInc
			) for j in xrange(rowNum) for i in xrange(colNum)
		]

	# rectangles used in the demo to see where you are looking
	def generateControlRectangles(self, rowNum, _colNum = None):
		colNum = rowNum if _colNum == None else _colNum
		xInc = self.canvasDim.x / (rowNum)
		yInc = self.canvasDim.y / (colNum)
		return [
			self.tracker.Rectangle(
				i * xInc, 
				j * yInc,
				xInc,
				yInc
			) for j in xrange(rowNum) for i in xrange(colNum)
		]

	def setCornerPointsInteractive(self):
		self.lookingPointMovAvg.setLength(15)

		points = self.generateCalibrationPoints(3)
		self.rects = self.generateControlRectangles(3)

		self.topLeft, _ = self.setPointAfterButton(
			circlePosition = points[0]
		)

		self.topCenter, _ = self.setPointAfterButton(
			circlePosition = points[1]
		)

		self.topRight, _ = self.setPointAfterButton(
			circlePosition = points[2]
		)

		self.middleLeft, _ = self.setPointAfterButton(
			circlePosition = points[3]
		)

		self.middleCenter, res = self.setPointAfterButton(
			circlePosition = points[4]
		)

		self.middleRight, _ = self.setPointAfterButton(
			circlePosition = points[5]
		)

		self.bottomLeft, _ = self.setPointAfterButton(
			circlePosition = points[6]
		)

		self.bottomCenter, _ = self.setPointAfterButton(
			circlePosition = points[7]
		)

		self.bottomRight, _ = self.setPointAfterButton(
			circlePosition = points[8]
		)

		# center relative to video size
		self.headCenter = self.getAverageEyePosition(res)

		self.xMax = (
			self.topRight.x + 
			self.middleRight.x + 
			self.bottomRight.x
		) / 3

		self.xMin = (
			self.topLeft.x + 
			self.middleLeft.x + 
			self.bottomLeft.x
		) / 3

		self.yMax = (
			self.bottomCenter.y + 
			self.bottomLeft.y + 
			self.bottomRight.y
		) / 3

		self.yMin = (
			self.topCenter.y + 
			self.topLeft.y + 
			self.topRight.y
		) / 3

		colorBounds = Point(0, 0)
		numTrials = 0
		for k in self.learningColors.keys():
			colorBounds += reduce(
				lambda p1, p2: p1 + p2, 
				self.learningColors[k]
			)
			numTrials += len(self.learningColors[k])

		avgColorMax = colorBounds.x / numTrials
		avgColorMin = colorBounds.y / numTrials

		print avgColorMin, avgColorMax

		self.tracker.setPupilTrained(avgColorMax, avgColorMin)

	def getAverageEyePosition(self, res):
		try:
			return Point( 
				int(
					np.mean(
						map(
							lambda eye: self.getCenter(
								eye.getHaarRectangle()
							).x,
							res
						)
					)
				),
				int(
					np.mean(
						map(
							lambda eye: self.getCenter(
								eye.getHaarRectangle()
							).y,
							res
						)
					)
				)
			)
		except ValueError:
			return Point(0, 0)

	def calibrate(self):
		self.setCornerPointsInteractive()
		self.run()

	def run(self):
		self.lookingPointMovAvg.setLength(8)
		self.setPointAfterButton(27, isLearning = False)

if __name__ == "__main__":
	np.seterr(all='ignore')
	eyeCalibration = EyeCalibration()
	eyeCalibration.calibrate()

