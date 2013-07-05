
import cv2
import sys
import numpy as np
import eyetracker.blob as blob

cv2.namedWindow('Display Window')
COLOR_MIN = np.array([0, 0, 0], np.uint8)
COLOR_MAX = np.array([0, 0, 0], np.uint8)
padding = 0
blobSize = 0

def printStats():
	print "Blob Threshold:", blobSize
	print "Range:", padding
	print "HSV:", COLOR_MAX

def changedBlobSize(thresh):
	global blobSize
	blobSize = thresh
	changedRange(padding)

def changedRange(nRange):
	global padding
	padding = nRange
	changed(0)(COLOR_MAX[0])
	changed(1)(COLOR_MAX[1])
	changed(2)(COLOR_MAX[2])
	printStats()

def changed(channel):
	def changedChannel(thresh):
		COLOR_MAX[channel] = thresh
		COLOR_MIN[channel] = thresh - padding if thresh - padding > 0 else 0
		global img, img_hsv
		img_disp = np.copy(img)
		img_thresh = cv2.inRange(img_hsv, COLOR_MIN, COLOR_MAX)
		bList = blob.getBlobs(img_thresh, blobSize, 10000000)
		cv2.drawContours(img_disp, map(lambda b: b.getContour(), bList), -1, (255, 0, 0), -1)
		#for b in bList:
			#cv2.circle(img_disp, b.getCentroid(), 10, (0,0,255), 2)
		cv2.imshow('Display Window', img_disp)
		printStats()
	return changedChannel

def main():
	if len(sys.argv) != 2:
	    print "Usage : python display_image.py <image_file>"

	else:
		global img, img_hsv
		img = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_COLOR)

		if (img == None):
			print "Could not open or find the image"
		else:
			img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			img_thresh = cv2.inRange(img, COLOR_MIN, COLOR_MAX)

			cv2.createTrackbar('Hue:','Display Window', 0, 255, changed(0))
			cv2.createTrackbar('Saturation:','Display Window', 0, 255, changed(1))
			cv2.createTrackbar('Value:','Display Window', 0, 255, changed(2))
			cv2.createTrackbar('Blob', 'Display Window', 0, 50000, changedBlobSize)
			cv2.createTrackbar('Range:', 'Display Window', 0, 255, changedRange)

			cv2.imshow('Display Window',img)

			print "size of image: ", img.shape
			cv2.waitKey(0)
			cv2.destroyAllWindows()

if __name__ == "__main__":
	main()

