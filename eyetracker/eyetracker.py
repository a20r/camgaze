
import cv2
import blob
import numpy as np
from collections import namedtuple
from eyestats import EyeStats
from trackingstats import TrackingStats
from point import Point

class EyeTracker:

	def __init__(self, img_input = None):

		self.Rectangle = namedtuple("Rectangle", "x y w h")

		if img_input != None:
			self.img_input = img_input
			self.img_orig = cv2.resize(
				img_input, 
				(self.xScale, self.yScale)
			)

		# detects the eye using haar cascades
		self.cascade = cv2.CascadeClassifier(
			"cascades/haarcascade_eye_tree_eyeglasses.xml"
		)

		# pupil color constants
		self.pupilThresh = 9000

		# the resized width and height for analysis
		self.xScale = 640
		self.yScale = 480

		self.padding = 10

		# found empircally
		self.averageContourSize = 10000

		self.MAX_COLOR = 60
		self.MIN_COLOR = 0

		self.previousEyes = list()
		self.lostEyes = set()

	def setImage(self, image):
		self.img_input = image
		self.img_orig = cv2.resize(
			self.img_input, 
			(self.xScale, self.yScale)
		)
		return self

	def _val2np(self, val):
		return np.array([val], np.uint8)

	def _tupleSum(self, t1, t2):
		retList = list()
		for i in xrange(len(t1)):
			retList += [t1[i] + t2[i] / 1]
		return tuple(retList)

	def mapVal(self, x, in_min, in_max, out_min, out_max):
	  return (
	  	(x - in_min) * 
	  	(out_max - out_min) / 
	  	(in_max - in_min) + 
	  	out_min
	  )

	def getAngle(self, P1, P2):
		deltaY = P2[1] - P1[1]
		deltaX = P2[0] - P1[0]
		angleInDegrees = np.arctan(
			float(deltaY) / float(deltaX)
		) * 180 / np.pi
		#print angleInDegrees
		return angleInDegrees

	# takes in a tuple point NOT A BLOB
	def getAverageAngle(self, p):
		cornerList = [(0, 0), (self.xScale, 0), 
			(0, self.yScale), (self.xScale, self.yScale)]
		return np.mean(
			map(
				lambda corner: abs(
					self.getAngle(p, corner)
				), 
				cornerList
			)
		)

	def weightPupil(self, possiblePupil):
		angleDev = abs(
			self.getAverageAngle(
				possiblePupil.getCentroid()
			) - 22.5
		)
		sizeDev = abs(
			possiblePupil.getContourArea() - 
			self.averageContourSize
		)
		return angleDev * sizeDev

	def filterAngle(self, p1, p2, padding):
		return (
			abs(self.getAngle(p1, p2)) < padding or
			abs(self.getAngle(p1, p2)) - 45 < padding
		)

	def filterBlobs(self, blobList, padding):
		cornerList = [
			(0, 0), 
			(self.xScale, 0), 
			(0, self.yScale), 
			(self.xScale, self.yScale)
		]
		return filter(
			lambda b: all(
				self.filterAngle(
					b.getCentroid(), 
					corner, 
					self.padding
				) for corner in cornerList
			), 
			blobList
		)

	def getPupil(self, img):
		possiblePupils = list()
		step = 10

		for minColor in xrange(
			self.MIN_COLOR, 
			self.MAX_COLOR - step, step):
			for maxColor in xrange(
				minColor + step, 
				self.MAX_COLOR, step):
				pPupil = self.getUnfilteredPupil(
					img, 
					minColor, 
					maxColor
				)
				if pPupil != None:
					possiblePupils.append(pPupil)

		if len(possiblePupils) == 0:
			return None

		return reduce(
			lambda p1, p2: 
			p1 if self.weightPupil(p1) < self.weightPupil(p2) 
			else p2, 
			possiblePupils
		)

	def getUnfilteredPupil(self, img, minColor, maxColor):

		# creates a binary image via color segmentation
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.equalizeHist(img)
		pupilBW = cv2.inRange(
			img, 
			self._val2np(minColor), 
			self._val2np(maxColor)
		)
		pupilBList = blob.getBlobs(pupilBW, self.pupilThresh)
		pupilBList = self.filterBlobs(pupilBList, self.padding)

		if len(pupilBList) == 0:
			return None

		maxIndex = [
			i for i, j in enumerate(pupilBList) 
			if j.getContourArea() == max(
				map(
					lambda pupil: pupil.getContourArea(), 
					pupilBList
				)
			)
		][0]
		return pupilBList[maxIndex]

	def drawPupils(self, img_centroid, pb, eyeStats):
		eyeStats.setPupil(pb)
		cv2.circle(
			img_centroid, 
			pb.getCentroid().toTuple(), 
			10, 
			(0, 255, 255), 
			20
		)
		cv2.line(
			img_centroid, 
			pb.getCentroid().toTuple(), 
			(0, 0), 
			(255, 255, 255), 
			10
		)
		cv2.line(
			img_centroid, 
			pb.getCentroid().toTuple(), 
			(
				img_centroid.shape[1], 
				img_centroid.shape[0]
			), 
			(255, 255, 255), 
			10
		)
		cv2.line(
			img_centroid, 
			pb.getCentroid().toTuple(), 
			(
				0, 
				img_centroid.shape[0]
			), 
			(255, 255, 255), 
			10
		)
		cv2.line(
			img_centroid, 
			pb.getCentroid().toTuple(), 
			(
				img_centroid.shape[1], 
				0
			), 
			(255, 255, 255), 
			10
		)

	def getXScale(self):
		return self.xScale

	def getYScale(self):
		return self.yScale

	def getRectSizes(self, rects):
		return map(lambda rect: rect[2] * rect[3], rects)

	def filterRectSize(self, rects):
		try:
			W, H = [(w, h) for _, _, w, h in rects 
				if w * h == min(self.getRectSizes(rects))][0]
			#print maxW, maxH
			return [
				(
					x + (w / 2) - (W / 2), 
					y + (h / 2) - (H / 2), 
					W, 
					H
				) for x, y, w, h in rects
			]
		except IndexError:
			return list()

	def matchFace(self, eyeStats, faceRects):
		hr = eyeStats.getHaarRectangle()
		haarCentroid = Point(
			hr.x + hr.w / 2, 
			hr.y + hr.h / 2
		)
		for x, y, w, h in faceRects:
			if haarCentroid.x <= x + w and \
				haarCentroid.x >= x and \
				haarCentroid.y <= y + h and \
				haarCentroid.y >= y:
				eyeStats.setFace(self.Rectangle(x, y, w, h))
				return

	def track(self):
		trackingStats = TrackingStats()

		img = np.copy(self.img_orig)
		trackingStats.setImage(np.copy(img))

		# uses Haar classification to find the eyes
		unfilteredEyeRects = self.cascade.detectMultiScale(
			cv2.cvtColor(
				img, 
				cv2.COLOR_BGR2GRAY
			), 
			scaleFactor = 2.2, 
			minNeighbors = 4, 
			maxSize = (200, 200), 
			minSize = (0, 0)
		)

		self.eyeRects = self.filterRectSize(unfilteredEyeRects)

		img_disp_colors = np.copy(self.img_orig)
		img_disp_centroids = np.copy(self.img_orig)
		img_disp_tracking = np.copy(self.img_orig)

		# changes the ROI of the image
		for x, y, w, h in self.eyeRects:
			eyeStats = EyeStats()
			eyeStats.setHaarRectangle(self.Rectangle(x, y, w, h))

			img = self.img_orig[y:y+h, x:x+w]
			eyeStats.setImage(self.img_orig[y:y+h, x:x+w])
			img = cv2.resize(img, (self.xScale, self.yScale))

			img_centroid = np.copy(img)
			img_tracking = np.copy(img)

			pupil = self.getPupil(img)
			if pupil == None:
				continue

			# draws the pupil onto the image
			cv2.drawContours(
				img, 
				[pupil.getContour()],
				 -1, 
				 (0, 255, 255), 
				 -1
			)

			# draws a circle for the pupils centroids found
			self.drawPupils(img_centroid, pupil, eyeStats)

			resVec = eyeStats.getResultantVector(self.xScale, self.yScale)
			# draws the centroid on the tracking image
			cv2.circle(
				img_tracking, 
				eyeStats.getPupil().getCentroid().toTuple(), 
				10, 
				(0, 255, 255), 
				20
			)

			# gets the end point and maps it to be visualized
			px, py = eyeStats.getPupil().getCentroid() + resVec
			endPoint = (px, py)

			# draws lines indicating looking position
			cv2.line(
				img_tracking, 
				eyeStats.getPupil().getCentroid().toTuple(),
				endPoint, 
				(0, 255, 0), 
				20
			)
			cv2.line(
				img_centroid, 
				eyeStats.getPupil().getCentroid().toTuple(),
				(eyeStats.getPupil().getCentroid() + resVec).toTuple(),
				(0, 255, 0), 
				20
			)

			# resizes images so they can be put back into the original image
			img = cv2.resize(img, (w, h))
			img_centroid = cv2.resize(img_centroid, (w, h))
			img_tracking = cv2.resize(img_tracking, (w, h))

			# updates the stats
			eyeStats.setTrackingImage(np.copy(img_tracking))
			eyeStats.setColorImage(np.copy(img))
			eyeStats.setCentroidImage(np.copy(img_centroid))

			# sets the inner ROIs to the colored and centroid images
			img_disp_colors[y:y+h, x:x+w] = img
			img_disp_centroids[y:y+h, x:x+w] = img_centroid
			img_disp_tracking[y:y+h, x:x+w] = img_tracking
			cv2.rectangle(
				img_disp_colors, 
				(x, y), 
				(x + w, y + h), 
				(255, 0, 0), 
				2
			)
			cv2.rectangle(
				img_disp_centroids, 
				(x, y), 
				(x + w, y + h), 
				(255, 0, 0), 
				2
			)
			trackingStats.pushEye(eyeStats)

		trackingStats.setColorImage(np.copy(img_disp_colors))
		trackingStats.setTrackingImage(np.copy(img_disp_tracking))
		trackingStats.setCentroidImage(np.copy(img_disp_centroids))

		self.lostEyes.update(trackingStats.assignIds(self.previousEyes))
		self.previousEyes = trackingStats.getEyeList() + list(self.lostEyes)

		return trackingStats

	def norm(self, p1, p2):
		"""
		Finds the distance between the two points
		"""
		return np.sqrt(
			pow(p1[0] - p2[0], 2) + 
			pow(p1[1] - p2[1], 2)
		)
