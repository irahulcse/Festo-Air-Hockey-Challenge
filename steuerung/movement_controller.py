import math
from steuerung.UDP_connector import UDPConnector

class MovementController:
    def __init__(self, striker_speed=850): #50% override
        self.striker_speed = striker_speed  # mm/s or your unit

    def _normalize(self, v):
        mag = math.hypot(*v)
        return (v[0]/mag, v[1]/mag) if mag else (0.0, 0.0)

    def _field_contains(self, x, y):
        return -5 <= x <= 481 and -4 <= y <= 483

    def compute_direction_and_speed(self, puck_current, puck_previous):
        dx = puck_current[0] - puck_previous[0]
        dy = puck_current[1] - puck_previous[1]
        dt = 10.0 / 300.0
        speed = math.hypot(dx, dy) / dt
        return speed, self._normalize((dx, dy))

    def find_reachable_intercept_point(self, puck_current, puck_previous, striker_position, max_time=3.0):
        speed, direction = self.compute_direction_and_speed(puck_current, puck_previous)

        for step in range(1, int(max_time * 10) + 1):
            t = step * 0.1
            fx = puck_current[0] + direction[0] * speed * t
            fy = puck_current[1] + direction[1] * speed * t

            if not self._field_contains(fx, fy):
                continue

            dist = math.hypot(fx - striker_position[0], fy - striker_position[1])
            safety_factor = 0.9  #Account for delay
            if dist / self.striker_speed <= t * safety_factor:
                return (fx, fy), direction, t

        return None, None, None

    def predict_defense_intercept(self, puck_current, puck_previous):
        dx = puck_current[0] - puck_previous[0]
        dy = puck_current[1] - puck_previous[1]
        goal_line_x = 150.0
        dt = 10.0 / 300.0
        vx = dx / dt
        vy = dy / dt

        if abs(vx) < 1e-5:
            predicted_y = puck_current[1]
        else:
            time_to_goal = (goal_line_x - puck_current[0]) / vx
            predicted_y = puck_current[1] + vy * time_to_goal

        predicted_y = max(-4, min(483, predicted_y))
        return (goal_line_x, predicted_y)

    def run_decision_step(self, puck_current, puck_previous, striker_position):
        reachable, direction, t = self.find_reachable_intercept_point(
            puck_current, puck_previous, striker_position)

        if reachable:
            print(f'Intercept at {reachable} in {t:.2f}s, dir {direction}')
            target = reachable
        else:
            print('No reachable point found. Switching to defense mode.')
            target = self.predict_defense_intercept(puck_current, puck_previous)

        with UDPConnector('192.168.4.201', 3001) as plc:
            #plc.setpoints(velocity=1.2, acceleration=0.8)
            plc.send_coordinates(*target)
            plc.send_coordinates(10,230)


