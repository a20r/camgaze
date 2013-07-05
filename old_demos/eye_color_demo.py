
import cv2
import sys
import eyetracker

def toggleView(thresh):
	"""
	Toggles the picture being shown
	"""
	global img_disp_colors, img_disp_centroids
	if thresh == 0:
		cv2.imshow('Color Segmentation', results.getColorImage())
	else:
		cv2.imshow('Color Segmentation', results.getCentroidImage())

def main():
	if len(sys.argv) != 2:
	    print "Usage : python display_image.py <image_file>"
	else:
		cv2.namedWindow('Color Segmentation')

		img_orig = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_COLOR)

		tracker = eyetracker.EyeTracker(img_orig)
		global results
		results = tracker.track()
		#print results

		cv2.createTrackbar('Toggle View', 'Color Segmentation', 0, 1, toggleView)

		cv2.imshow('Color Segmentation', results.getColorImage())

		cv2.waitKey(0)
		cv2.destroyAllWindows()
		return True

if __name__ == "__main__":
	main()
