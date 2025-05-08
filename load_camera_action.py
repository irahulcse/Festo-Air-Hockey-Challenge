import numpy as np
import cv2 as cv
from coordinates.coordinates_transformation import calc_robot_coordinates
from coordinates.coordinates_memory import CoordinateBuffer

def load_camera_action(memory):

    cap = cv.VideoCapture(2)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    for i in range(2):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_no_noise = cv.GaussianBlur(gray, (9,9), 2)


        rows = gray.shape[0]
        circles = cv.HoughCircles(gray_no_noise, cv.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=30,
                                   minRadius=30, maxRadius=40)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                #print(center)
                robot_x, robot_y = calc_robot_coordinates(center[0], center[1])
                #print(robot_x, robot_y)
                memory.add(robot_x, robot_y)
                # circle center
                cv.circle(gray, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(gray, center, radius, (255, 0, 255), 3)



        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    #cv.destroyAllWindows()
    return memory