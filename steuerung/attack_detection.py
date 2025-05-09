# attack_detection.py
import numpy as np

def detect_opponent_field_attack(storage):
    """Detects potential attack based on direction change in opponent zone."""
    three_latest_points = storage.get_latest_puck_points(3)
    x_array = three_latest_points[:, 0]
    y_array = three_latest_points[:, 1]

    if len(three_latest_points) < 3:
        return False  # Not enough data to detect direction change
    
    # Direction change in x or y
    if np.sign(x_array[0] - x_array[1]) != np.sign(x_array[1] - x_array[2]):
        return True
    if np.sign(y_array[0] - y_array[1]) != np.sign(y_array[1] - y_array[2]):
        return True

    return False

def detect_own_field_attack(puck_current, puck_previous):
    """Detects if puck is moving toward own goal."""
    dx = puck_current[0] - puck_previous[0]
    dt = 10.0 / 300.0
    vx = dx / dt
    return vx < 0

def detect_attack(storage, puck_current, puck_previous, mid_field_x=350.0):
    """Decides which detection method to use based on puck's zone."""
    if puck_current[0] > mid_field_x:
        return detect_opponent_field_attack(storage)
    else:
        return detect_own_field_attack(puck_current, puck_previous)


# test_attack_and_movement.py
from movement_controller import MovementController
import attack_detection
import matplotlib.pyplot as plt
import numpy as np
import time

class MockStorage:
    def __init__(self, puck_positions):
        self.puck_positions = puck_positions

    def get_latest_puck_points(self, n):
        return np.array(self.puck_positions[-n:])

def main():
    controller = MovementController()
    puck_path = [
        (400, 200), (390, 210), (380, 220),
        (370, 230), (360, 240), (350, 250),
        (340, 260), (330, 270), (320, 280)
    ]
    striker_position = (350.0, 241.5)
    storage = MockStorage([])

    puck_x, puck_y = [], []
    striker_x, striker_y = [], []

    for i in range(1, len(puck_path)):
        puck_previous = puck_path[i - 1]
        puck_current = puck_path[i]
        storage.puck_positions.append(puck_current)

        print(f"\nFrame {i}: Puck prev={puck_previous}, current={puck_current}")

        attack = attack_detection.detect_attack(
            storage, puck_current, puck_previous)

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

        striker_position = puck_current
        time.sleep(0.1)

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(puck_x, puck_y, 'bo-', label='Puck path')
    plt.plot(striker_x, striker_y, 'ro--', label='Striker path')
    plt.axvline(350, color='gray', linestyle='--', label='Defense line (x=350)')
    plt.xlabel("x (bottom to top)")
    plt.ylabel("y (right to left)")
    plt.title("Puck and Striker Trajectory with Attack Detection")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    main()
