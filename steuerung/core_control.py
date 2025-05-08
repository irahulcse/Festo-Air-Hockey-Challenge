from UDP_connector import UDPConnector
import math

class CoreControl():
 def __init__(self):
  self.last_puck_position = (0.0, 0.0)
  self.striker_position = (350, 50)  # Aktuelle Position des Schlägers
  self.max_speed = 1700.0  # mm/s
  self.striker_speed = self.max_speed * 0.5  # mm/s
  
 def compute_direction_and_speed(self, puck_position):
        dx = puck_position[0] - self.last_puck_position[0]
        dy = puck_position[1] - self.last_puck_position[1]
        dt = 10/300  # Zeit zwischen Positionsmessungen in Sekunden
        speed = math.sqrt(dx**2 + dy**2) / dt
        direction = self.normalize((dx, dy))
        return speed, direction

 def is_inside_field(self, x, y):
    return -5 <= x <= 481 and -4 <= y <= 483
 
 def normalize(self, v):
        mag = math.sqrt(v[0]**2 + v[1]**2)
        return (v[0]/mag, v[1]/mag) if mag != 0 else (0, 0)

 def find_reachable_intercept_point(self, current_puck_pos, max_time=3.0):
    puck_speed, direction = self.compute_direction_and_speed(current_puck_pos)

    for step in range(1, int(max_time * 10)):  # Schritte von 0.1s bis max_time
        t = step * 0.1
        puck_future_x = current_puck_pos[0] + direction[0] * puck_speed * t
        puck_future_y = current_puck_pos[1] + direction[1] * puck_speed * t
        print("x:", puck_future_x, "y:", puck_future_y)
        
        if not self.is_inside_field(puck_future_x, puck_future_y):
            continue
        
        distance = math.sqrt(
            (puck_future_x - self.striker_position[0])**2 +
            (puck_future_y - self.striker_position[1])**2
        )
        print(distance)
        time_needed = distance / self.striker_speed
        if time_needed <= t:
            return (puck_future_x, puck_future_y), direction, t  # Punkt ist erreichbar

    return None, None, None  # Kein erreichbarer Punkt

 def compute_return_direction(self, incoming_direction):
    return (-incoming_direction[0], -incoming_direction[1]) 

last_puck_position = (450, 80)
current_puck_position = (450, 80)
mallet_pos = (350, 50)

core_control = CoreControl()
core_control.last_puck_position = last_puck_position
reachable_point, direction, time = core_control.find_reachable_intercept_point(current_puck_position)

udp_connector = UDPConnector()
udp_connector.send_coordinates()(reachable_point[0], reachable_point[1])

if reachable_point:
    print(f"Erreichbarer Punkt: {reachable_point}, Richtung: {direction}, Zeit: {time}")
else:
    print("Kein erreichbarer Punkt gefunden.")