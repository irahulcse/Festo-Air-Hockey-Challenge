import numpy as np

from coordinates.coordinates_memory import CoordinateBuffer


def detect_attack_from_opponent(memory: CoordinateBuffer):
    three_latest_points = memory.latest(3)
    if not three_latest_points or len(three_latest_points) < 3:
        return False

    three_latest_points = np.array(three_latest_points)

    x_array = three_latest_points[:,0]
    y_array = three_latest_points[:,1]

    # Richtungsänderung in x-Richtung
    if np.sign((x_array[0] - x_array[1])) < np.sign((x_array[1] - x_array[2])):
        return True

    return False