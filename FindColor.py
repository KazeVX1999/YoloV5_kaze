import cv2
import numpy as np
import time

# This python file run for finding the exact colour.
# Code Adapted from shashiben, 2020.

def fun(x):
    pass

# Capture Video
video = cv2.VideoCapture(0)

# Window for trackbar
cv2.namedWindow("Detection")
cv2.createTrackbar("LH", "Detection", 0, 255, fun)
cv2.createTrackbar("LS", "Detection", 0, 255, fun)
cv2.createTrackbar("LV", "Detection", 0, 255, fun)

cv2.createTrackbar("UH", "Detection", 180, 180, fun)
cv2.createTrackbar("US", "Detection", 255, 255, fun)
cv2.createTrackbar("UV", "Detection", 255, 255, fun)
enterBGR = (42, 181, 26)
enter = [[10,10], [100, 100]]

exitBGR = (16, 23, 233)
exit =[[120, 120],[190, 190]]


from ApiOperator import ApiOperator

apiOperator = ApiOperator()
apiOperator.loginCamera()
apiOperator.extractMarks()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

while True:
    t1 = time.time()
    # Read boolean and frame
    ret, image = camera.read()

    for SetMarks in apiOperator.cameraEnterMarks:
        image = cv2.line(image, (SetMarks[0][0], SetMarks[0][1]), (SetMarks[1][0], SetMarks[1][1]), color=(42, 181, 26),
                         thickness=8)
    for SetMarks in apiOperator.cameraExitMarks:
        image = cv2.line(image, (SetMarks[0][0], SetMarks[0][1]), (SetMarks[1][0], SetMarks[1][1]), color=(16, 23, 233),
                         thickness=8)

    # Convert it into HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Handle Keyboard events
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord("s"):
        cv2.imwrite("Image.jpeg", image)
        break

    # Get positions of trackbars
    lh = cv2.getTrackbarPos("LH", "Detection")
    ls = cv2.getTrackbarPos("LS", "Detection")
    lv = cv2.getTrackbarPos("LV", "Detection")

    uh = cv2.getTrackbarPos("UH", "Detection")
    us = cv2.getTrackbarPos("US", "Detection")
    uv = cv2.getTrackbarPos("UV", "Detection")

    # Lower bound of color
    lower_bound = np.array([lh, ls, lv])

    # Higher bound of color
    upper_bound = np.array([uh, us, uv])

    # create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Bitwise And operation
    result = cv2.bitwise_and(image, image, mask=mask)


    # Display normal frame
    cv2.imshow("Frame", image)

    # Display the mask
    cv2.imshow("Mask", mask)

    # Display only the needed color
    cv2.imshow("Result", result)


# Release the video
video.release()

# Destroy the windows
cv2.destroyAllWindows()

# End of Code Adapted