# attack_detection.py
import numpy as np

from coordinates.coordinates_memory import CoordinateBuffer


def detect_attack_from_opponent(memory: CoordinateBuffer):
    three_latest_points = memory.latest(3)
    if not three_latest_points or len(three_latest_points) < 3:
        return False

    three_latest_points = np.array(three_latest_points)

    x_array = three_latest_points[:, 0]
    y_array = three_latest_points[:, 1]

    # Richtungsänderung in x-Richtung
    if np.sign((x_array[0] - x_array[1])) < np.sign((x_array[1] - x_array[2])):
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
        return detect_attack_from_opponent(storage)
    else:
        return detect_own_field_attack(puck_current, puck_previous)


# test_attack_and_movement.py
import numpy as np

