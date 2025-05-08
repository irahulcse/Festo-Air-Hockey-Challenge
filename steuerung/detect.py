from storage import Storage
import numpy as np

def detect_attack_from_opponent(storage):
    three_latest_points = storage.get_latest_puck_points(3)
    
    x_array = three_latest_points[:,0]
    y_array = three_latest_points[:,1]

    # Richtungsänderung in x-Richtung
    if np.sign((x_array[0]-x_array[1])) != np.sign((x_array[1]-x_array[2])):
        return True
    
    # Richtungsänderung in y-Richtung
    if np.sign(y_array[0]-y_array[1]) != np.sign(y_array[1]-y_array[2]):
        return True

    return False