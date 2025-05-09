import numpy as np

from coordinates import coordinates_transformation
from coordinates.coordinates_memory import CoordinateBuffer
from steuerung.detect import detect_attack_from_opponent
import time
import cv2 as cv

def main():

    memory = CoordinateBuffer()
 #   controller = CoreControl()

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_no_noise = cv.GaussianBlur(gray, (9, 9), 2)

        rows = gray.shape[0]
        circles = cv.HoughCircles(gray_no_noise, cv.HOUGH_GRADIENT, 1, rows / 8,
                                  param1=100, param2=30,
                                  minRadius=30, maxRadius=40)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                print(center)
                robot_x, robot_y = coordinates_transformation.calc_robot_coordinates(center[0], center[1])
                print(robot_x, robot_y)
                memory.add(robot_x, robot_y)
                # circle center
                cv.circle(gray, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(gray, center, radius, (255, 0, 255), 3)
                # Angriff erkennen
                if detect_attack_from_opponent(memory):
                    print("Angriff erkannt!")
                    time.sleep(3)  # 30–40 FPS Ziel

                    # Reaktionslogik: Blocken?
                    puck_pos = memory.latest(1)
            #                if puck_pos:
            #                    intercept, direction, t = controller.find_reachable_intercept_point(puck_pos[0])
            #                    if intercept:
            #                        print(f"Abwehr bei {intercept} in {t:.2f}s")
            # UDP senden

        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()




if __name__ == "__main__":
    main()
