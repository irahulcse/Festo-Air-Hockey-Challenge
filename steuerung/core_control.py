from UDP_connector import UDPConnector

class CoreControl():
 def __init__(self):
  #self.current_position = (0.0, 0.0)
  self.last_position = (0.0, 0.0)
  
def compute_velocity_and_direction(x, y):
    # Compute the velocity and direction based on the current position
    # For example, you can use the difference between the current and last positions
    delta_x = x - self.last_position[0]
    delta_y = y - self.last_position[1]

    time = 10/3 
    
    # Calculate velocity and direction
    velocity = ((delta_x**2 + delta_y**2)**0.5 ) / time 
    direction = (delta_x, delta_y)  # Direction vector
    
    return velocity, direction


def compute_attack_position(self, x,y ):
    # Compute the attack position based on the current position
    velocity, direction = self.compute_velocity_and_direction(x, y)
    attack_x = x + direction[0] * velocity
    attack_y = y + direction[1] * velocity
    
    return (attack_x, attack_y)