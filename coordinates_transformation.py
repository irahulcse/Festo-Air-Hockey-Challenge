def calc_robot_coordinates(x, y):
    x_robot = (x - 290)/728  * 450 + 10
    y_robot = (y - 164)/800  * 450 + 10
    return x_robot, y_robot