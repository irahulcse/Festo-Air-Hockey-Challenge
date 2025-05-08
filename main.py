from coordinates.coordinates_memory import CoordinateBuffer 
from steuerung.environment import Environment


if __name__ == "__main__":

    coordinateBuffer = CoordinateBuffer()
    environment = Environment()
    print(environment.goal_boundaries)
    