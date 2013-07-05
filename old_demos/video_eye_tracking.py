import cv2
import numpy as np
import eyetracker

c = cv2.VideoCapture(0)
tracker = eyetracker.EyeTracker()

while(1):
    _, f = c.read()
    tracker.setImage(f)
    results = tracker.track()
    #cv2.imshow('e2', cv2.inRange(cv2.cvtColor(results.getImage(), cv2.COLOR_BGR2GRAY), 
    	#np.array([20], np.uint8), np.array([40], np.uint8)))

    cv2.imshow('e2', results.getCentroidImage())

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()