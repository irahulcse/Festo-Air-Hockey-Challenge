from UDP_connector import UDPConnector
import math

class CoreControl():
 def __init__(self):
  #self.current_position = (0.0, 0.0)
  self.last_puck_position = (0.0, 0.0)
  self.striker_position = (0.0, 500.0)  # Aktuelle Position des Schlägers
  self.max_speed = 1700.0  # mm/s
  self.striker_speed = self.max_speed * 0.5  # mm/s
  
 def compute_direction_and_speed(self, puck_position):
        dx = puck_position[0] - self.last_puck_position[0]
        dy = puck_position[1] - self.last_puck_position[1]
        dt = 100/3  # Zeit zwischen Positionsmessungen in Sekunden
        speed = math.sqrt(dx**2 + dy**2) / dt
        direction = self.normalize((dx, dy))
        return speed, direction

 def is_inside_field(self, x, y):
    return -5 <= x <= 481 and -4 <= y <= 483

 def find_reachable_intercept_point(self, current_puck_pos, max_time=3.0):
    puck_speed, direction = self.compute_puck_motion(current_puck_pos)

    for step in range(1, int(max_time * 10)):  # Schritte von 0.1s bis max_time
        t = step * 0.1
        puck_future_x = current_puck_pos[0] + direction[0] * puck_speed * t
        puck_future_y = current_puck_pos[1] + direction[1] * puck_speed * t
        if not self.is_inside_field(puck_future_x, puck_future_y):
            continue
        
        distance = math.sqrt(
            (puck_future_x - self.striker_position[0])**2 +
            (puck_future_y - self.striker_position[1])**2
        )

        time_needed = distance / self.striker_speed
        if time_needed <= t:
            return (puck_future_x, puck_future_y), direction, t  # Punkt ist erreichbar

    return None, None, None  # Kein erreichbarer Punkt

 def compute_return_direction(self, incoming_direction):
    return (-incoming_direction[0], -incoming_direction[1]) 

last_puck_position = (470, 30)
current_puck_position = (490, 10)
mallet_pos = (350, 50)

core_control = CoreControl()
core_control.last_puck_position = last_puck_position
reachable_point, direction, time = core_control.find_reachable_intercept_point(current_puck_position)

if reachable_point:
    print(f"Erreichbarer Punkt: {reachable_point}, Richtung: {direction}, Zeit: {time}")
else:
    print("Kein erreichbarer Punkt gefunden.")