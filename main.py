from coordinates.coordinates_memory import CoordinateBuffer 
from steuerung.environment import Environment
from load_camera_action import load_camera_action
from steuerung.UDP_connector import UDPConnector




if __name__ == "__main__":

    coordinateBuffer = CoordinateBuffer()
    environment = Environment()
    load_camera_action()
    print(environment.goal_boundaries)
    print(coordinateBuffer.latest(2))
    #print(x,y)
    positions = load_camera_action()
    print(positions)
    #UDPConnector.send_coordinates(x, y)
