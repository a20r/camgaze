
import uuid

class TrackingStats(object):

	def __init__(self):
		self.centroidImage = None
		self.image = None
		self.colorImage = None
		self.trackingImage = None
		self.eyeList = list()
		self.idMap = dict()

	# assigns unique identifiers to pupils
	# returns the eyes that are no longer in the frame
	# takes a list of eyes from the last frame
	# and the lost eyes
	def assignIds(self, prevEyes):
		if len(self.eyeList) == 0:
			return set()

		if len(prevEyes) == 0:
			for eye in self.eyeList:
				eyeId = str(uuid.uuid4())
				self.idMap[eyeId] = eye
				eye.setId(eyeId)
			return set()
		else:
			distList = [
				enumerate(
					[
						eye.norm(
							eye.getHaarCentroid(),
							pEye.getHaarCentroid()
						) for pEye in prevEyes
					]
				) for eye in self.eyeList
			]

			minDistList = map(
				lambda ds: reduce(
					lambda a, b: a if a[1] < b[1] else b, ds
				),
				distList
			)
			usedPrevEyes = list()
			for i, (j, _) in enumerate(minDistList):
				if self.eyeList[i].getId() == None and not j in usedPrevEyes:
					self.eyeList[i].setId(prevEyes[j].getId())
					self.idMap[prevEyes[j].getId()] = self.eyeList[i]
					usedPrevEyes += [j]

			for i in xrange(len(minDistList)):
				if self.eyeList[i].getId() == None:
					eyeId = str(uuid.uuid4())
					self.eyeList[i].setId(eyeId)
					self.idMap[eyeId] = self.eyeList[i]

			return set(prevEyes) - set(prevEyes[j] for j in usedPrevEyes)

	def pushEye(self, eye):
		self.eyeList += [eye]
		return self

	def setColorImage(self, cImage):
		self.colorImage = cImage
		return self

	def setCentroidImage(self, cImage):
		self.centroidImage = cImage
		return self

	def setImage(self, image):
		self.image = image
		return self

	def setTrackingImage(self, image):
		self.trackingImage = image
		return self

	def getTrackingImage(self):
		return self.trackingImage

	def getColorImage(self):
		return self.colorImage

	def getCentroidImage(self):
		return self.centroidImage

	def getImage(self):
		return self.image

	def getEye(self, index):
		return self.eyeList[index]

	def getEyeList(self):
		return self.eyeList

	def __str__(self):
		retString = str()
		for eye in self.eyeList:
			retString += str(eye) + " "
		return retString

	def __len__(self):
		return len(self.eyeList)

	def __getitem__(self, key):
		if type(key) == int:
			return self.eyeList[key]
		elif type(key) == str:
			return self.idMap[key]
		else:
			raise TypeError(
				"Cannot index the class using type: " +
				type(key)
			)