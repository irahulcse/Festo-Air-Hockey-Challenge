from coordinates.coordinates_memory import CoordinateBuffer
from load_camera_action import load_camera_action
from steuerung.detect import detect_attack_from_opponent
import time
import cv2 as cv

def main():

    memory = CoordinateBuffer()
 #   controller = CoreControl()

    while True:
        # Bild verarbeiten und ggf. Koordinaten in memory ablegen
        memory = load_camera_action(memory)

        # Angriff erkennen
        if detect_attack_from_opponent(memory):
            print("Angriff erkannt!")
            time.sleep(0.03)  # 30–40 FPS Ziel

            # Reaktionslogik: Blocken?
            puck_pos = memory.latest(1)
#                if puck_pos:
#                    intercept, direction, t = controller.find_reachable_intercept_point(puck_pos[0])
#                    if intercept:
#                        print(f"Abwehr bei {intercept} in {t:.2f}s")
                    # UDP senden



if __name__ == "__main__":
    main()
