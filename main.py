import numpy as np

from coordinates import coordinates_transformation
from coordinates.coordinates_memory import CoordinateBuffer
from steuerung.attack_detection import detect_attack
from steuerung.movement_controller import MovementController
import time
import cv2 as cv

def main():

    striker_position = (7,245)
    memory = CoordinateBuffer()
    controller = MovementController()

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
        # Detect circles in the image  
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

            puck_x, puck_y = [], []
            striker_x, striker_y = [], []
            # Get ladest puck position
            puck_current = memory.latest(1)[0]
            puck_previous = memory.latest(2)[0]
            # Check for attack
            attack = detect_attack(memory, puck_current, puck_previous)
            if attack:
                print("\u26a0\ufe0f Attack detected!")
            else:
                print("No attack.")

            controller.run_decision_step(
                puck_current, puck_previous, striker_position
            )

            puck_x.append(puck_current[0])
            puck_y.append(puck_current[1])
            striker_x.append(striker_position[0])
            striker_y.append(striker_position[1])
            time.sleep(0.1)

        # Display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()




if __name__ == "__main__":
    main()
