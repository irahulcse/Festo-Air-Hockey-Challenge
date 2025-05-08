from coordinates.coordinates_memory import CoordinateBuffer 
from steuerung.environment import Environment
from load_camera_action import load_camera_action
from steuerung.UDP_connector import UDPConnector




if __name__ == "__main__":

    coordinateBuffer = CoordinateBuffer()
    environment = Environment()
    memory = load_camera_action(coordinateBuffer)
    #print(environment.goal_boundaries)
    print(memory.get_all())
    #print(x,y)
    positions = coordinateBuffer.latest(2)
    #UDPConnector.send_coordinates(x, y)
