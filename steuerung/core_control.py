# control_loop.py
import math
from steuerung.core_control import UDPConnector        #   ← file from previous message

class CoreControl:
    def __init__(self):
        self.last_puck_position = (0.0, 0.0)
        self.striker_position   = (350.0, 50.0)         
        self.max_speed          = 1700.0                
        self.striker_speed      = 0.5 * self.max_speed  

    def _normalize(self, v):
        mag = math.hypot(*v)
        return (v[0]/mag, v[1]/mag) if mag else (0.0, 0.0)

    def _field_contains(self, x, y):
        return -5 <= x <= 481 and -4 <= y <= 483         

    def compute_direction_and_speed(self, puck_pos):
        dx = puck_pos[0] - self.last_puck_position[0]
        dy = puck_pos[1] - self.last_puck_position[1]
        dt = 10.0 / 300.0                                
        speed = math.hypot(dx, dy) / dt
        return speed, self._normalize((dx, dy))

    def find_reachable_intercept_point(self, puck_pos, max_time=3.0):
        speed, direction = self.compute_direction_and_speed(puck_pos)

        for step in range(1, int(max_time * 10) + 1):    
            t = step * 0.1
            fx = puck_pos[0] + direction[0] * speed * t
            fy = puck_pos[1] + direction[1] * speed * t

            if not self._field_contains(fx, fy):
                continue

            dist = math.hypot(fx - self.striker_position[0],
                              fy - self.striker_position[1])
            if dist / self.striker_speed <= t:
                return (fx, fy), direction, t            

        return None, None, None                          

    def compute_return_direction(self, incoming_direction):
        return (-incoming_direction[0], -incoming_direction[1])

if __name__ == '__main__':
    last_puck_position     = (450.0, 80.0)
    current_puck_position  = (450.0, 80.0)

    control = CoreControl()
    control.last_puck_position = last_puck_position

    reachable, direction, t = control.find_reachable_intercept_point(
        current_puck_position)

    if reachable:
        print(f'Intercept at {reachable} in {t:.2f}s, dir {direction}')

        # send to PLC
        with UDPConnector('192.168.4.201', 3001) as plc:
            plc.setpoints(velocity=1.2, acceleration=0.8)
            plc.send_coordinates(*reachable)
    else:
        print('No reachable point found.')
