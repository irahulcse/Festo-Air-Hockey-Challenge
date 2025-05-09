from storage import Storage
import numpy as np

def detect_attack_from_opponent(memory: CoordinateBuffer):
    three_latest_points = memory.latest(3)
    if not three_latest_points or len(three_latest_points) < 3:
        return False

    three_latest_points = np.array(three_latest_points)

    x_array = three_latest_points[:,0]
    y_array = three_latest_points[:,1]

    x = x_array[:, 0]

    dx1 = x[1] - x[0]
    dx2 = x[2] - x[1]

    if dx1 < -5 and dx2 < -5:
        return True

    return False