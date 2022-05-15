import cv2
import numpy as np

enterBGR = (42, 181, 26)
lowerHSVEnter = (62, 215, 180)
upperHSVEnter = (64, 219, 182)

exitBGR = (16, 23, 233)
lowerHSVExit = (0, 238, 232)
upperHSVExit = (1, 238, 233)

import cv2
import numpy as np

#image = cv2.imread("marks.png")

fileName = ""

image = cv2.imread(fileName)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, lowerHSVExit, upperHSVExit)

## slice the green
imask = mask>0
a = np.nonzero(imask)
print(len(a))
greenfinal = np.zeros_like(image, np.uint8)
greenfinal[imask] = image[imask]

count = np.sum(np.nonzero(mask))
print("count =", count)
#cv2.imshow("Original", image)
#cv2.imshow("HSV", hsv)
cv2.imshow("Filtered", greenfinal)
cv2.waitKey(0)

