import math
import numpy as np
import matplotlib.pyplot as plt

class AirHockeyDefender:
    def __init__(self, table_width, table_height, arm_speed, defense_line, max_arm_x=None):
        """
        Initialize air hockey defense system
        
        Parameters:
        table_width: width of playing field (mm)
        table_height: height of playing field (mm)
        arm_speed: maximum arm speed (mm/s)
        defense_line: base defense line on X axis (mm)
        max_arm_x: maximum arm position on X (if None, uses defense_line + 100)
        """
        self.table_width = table_width
        self.table_height = table_height
        self.arm_speed = arm_speed
        self.defense_line = defense_line
        self.max_arm_x = max_arm_x if max_arm_x else defense_line + 100
        self.safety_margin = 50  # minimum distance from walls
        
        # For visualization and debugging
        self.debug_mode = False
        self.trajectory_points = []
        
    def calculate_puck_vector(self, prev_pos, current_pos, dt):
        """
        Calculate puck velocity vector and direction
        
        Parameters:
        prev_pos: previous puck position (x1, y1)
        current_pos: current puck position (x2, y2)
        dt: time between measurements (sec)
        
        Returns:
        velocity: velocity vector (vx, vy) in mm/s
        speed: puck speed (mm/s)
        """
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        
        # Calculate velocity components
        vx = dx / dt
        vy = dy / dt
        velocity = (vx, vy)
        
        # Calculate scalar speed
        speed = math.sqrt(vx**2 + vy**2)
        
        return velocity, speed
    
    def predict_trajectory(self, puck_pos, velocity, max_time=3.0, time_step=0.01):
        """
        Predict puck trajectory considering wall bounces
        """
        x, y = puck_pos
        vx, vy = velocity
        
        trajectory = [(x, y)]
        current_time = 0
        
        while current_time < max_time:
            # Calculate new position
            next_x = x + vx * time_step
            next_y = y + vy * time_step
            
            # Check for side wall bounces
            if next_y < 0:
                # Calculate exact time to bounce
                bounce_time = -y / vy
                # Update position to bounce point
                x = x + vx * bounce_time
                y = 0
                # Update remaining time
                remaining_time = time_step - bounce_time
                # Change Y direction
                vy = -vy
                # Continue movement after bounce
                x = x + vx * remaining_time
                y = y + vy * remaining_time
            elif next_y > self.table_height:
                # Similar for top wall
                bounce_time = (self.table_height - y) / vy
                x = x + vx * bounce_time
                y = self.table_height
                remaining_time = time_step - bounce_time
                vy = -vy
                x = x + vx * remaining_time
                y = y + vy * remaining_time
            else:
                x = next_x
                y = next_y
            
            current_time += time_step
            trajectory.append((x, y))
            
            # Stop if puck goes beyond back line
            if x < 0 or x > self.table_width:
                break
                
        return trajectory
        
    def find_intercept_point(self, puck_pos, velocity, arm_pos):
        """Find puck interception point"""
        if velocity[0] >= 0 or (velocity[0] == 0 and velocity[1] == 0):
            return (puck_pos[0], puck_pos[1]), None, "moving_away"
            
        trajectory = self.predict_trajectory(puck_pos, velocity)
        self.trajectory_points = trajectory
        
        # Find intersection points with defense line
        intercept_candidates = []
        
        for i in range(1, len(trajectory)):
            prev_point = trajectory[i-1]
            current_point = trajectory[i]
            
            # If trajectory crosses defense line
            if prev_point[0] > self.defense_line and current_point[0] <= self.defense_line:
                # Linear interpolation for exact intersection
                ratio = (prev_point[0] - self.defense_line) / (prev_point[0] - current_point[0])
                y_intercept = prev_point[1] + ratio * (current_point[1] - prev_point[1])
                time_to_point = (i-1) * 0.01 + ratio * 0.01
                
                # Check field boundaries without hard values
                if 0 <= y_intercept <= self.table_height:
                    intercept_candidates.append(((self.defense_line, y_intercept), time_to_point))
        
        if not intercept_candidates:
            # Return to field center (regardless of dimensions)
            return (self.defense_line, self.table_height/2), None, None
            
        return intercept_candidates[0][0], intercept_candidates[0][1], None

    def calculate_optimal_position(self, puck_prev_pos, puck_current_pos, arm_pos, dt):
        """Calculate optimal arm position for puck interception"""
        velocity, speed = self.calculate_puck_vector(puck_prev_pos, puck_current_pos, dt)
        
        intercept_point, intercept_time, special_case = self.find_intercept_point(
            puck_current_pos, velocity, arm_pos
        )
        
        if special_case == "moving_away" or intercept_time is None:
            return (self.defense_line, puck_current_pos[1])

        if speed == 0:
            return (self.defense_line, puck_current_pos[1])

        # Determine puck movement trajectory
        dx = puck_current_pos[0] - puck_prev_pos[0]
        dy = puck_current_pos[1] - puck_prev_pos[1]
        
        # Find intersection point with defense line
        # Use triangle similarity to find Y coordinate
        if dx != 0:  # Avoid division by zero
            # Units of Y per unit of X
            slope = dy / dx
            # X distance from current position to defense line
            x_to_defense = puck_current_pos[0] - self.defense_line
            # Y change when moving to defense line
            y_change = slope * x_to_defense
            # Y coordinate of intersection point
            intercept_y = puck_current_pos[1] - y_change
        else:
            intercept_y = puck_current_pos[1]

        # Check if we can reach interception point
        target_x = self.defense_line
        target_y = intercept_y
        
        distance_to_target = math.hypot(
            target_x - arm_pos[0],
            target_y - arm_pos[1]
        )
        
        time_to_reach = distance_to_target / self.arm_speed
        
        # If we can't reach interception point, find closest achievable point on trajectory
        if time_to_reach > intercept_time:
            # Maximum distance arm can travel
            max_distance = self.arm_speed * intercept_time
            
            # Normalized direction vector
            direction_x = target_x - arm_pos[0]
            direction_y = target_y - arm_pos[1]
            direction_length = math.sqrt(direction_x**2 + direction_y**2)
            
            if direction_length > 0:
                # Move towards interception point at maximum possible distance
                ratio = max_distance / direction_length
                target_x = arm_pos[0] + direction_x * ratio
                target_y = arm_pos[1] + direction_y * ratio
        
        # Boundary check
        margin = 10
        target_y = max(margin, min(target_y, self.table_height - margin))
        target_x = max(self.defense_line - 2, min(target_x, self.max_arm_x))
        
        return (target_x, target_y)

    def visualize_scenario(self, puck_prev_pos, puck_current_pos, arm_pos, target_pos, trajectory=None):
        """
        Visualize scenario with current positions and predicted trajectory
        
        Parameters:
        puck_prev_pos: previous puck position
        puck_current_pos: current puck position
        arm_pos: current arm position
        target_pos: target arm position
        trajectory: list of predicted trajectory points (optional)
        """
        plt.figure(figsize=(12, 6))
        
        # Draw playing field
        plt.plot([0, self.table_width, self.table_width, 0, 0], 
                 [0, 0, self.table_height, self.table_height, 0], 'k-', linewidth=2)
        
        # Draw defense line
        plt.plot([self.defense_line, self.defense_line], [0, self.table_height], 'r--', linewidth=2)
        
        # Draw center line (if any)
        midline_x = self.table_width / 2
        plt.plot([midline_x, midline_x], [0, self.table_height], 'k:', linewidth=1)
        
        # Draw trajectory if provided
        if trajectory and len(trajectory) > 0:
            traj_x = [point[0] for point in trajectory]
            traj_y = [point[1] for point in trajectory]
            plt.plot(traj_x, traj_y, 'b-', alpha=0.5, label='Predicted trajectory')
            
            # Mark important points on trajectory
            # Start and end
            plt.plot(traj_x[0], traj_y[0], 'bo', alpha=0.5)
            plt.plot(traj_x[-1], traj_y[-1], 'bo', alpha=0.5)
            
            # Mark bounce points if any
            for i in range(1, len(trajectory)-1):
                prev_dir = (trajectory[i][1] - trajectory[i-1][1])
                next_dir = (trajectory[i+1][1] - trajectory[i][1])
                if prev_dir * next_dir < 0:  # Change in Y direction
                    plt.plot(trajectory[i][0], trajectory[i][1], 'bo', markersize=8)
                    plt.text(trajectory[i][0]+10, trajectory[i][1]+10, f'Bounce', fontsize=9)
        
        # Draw puck velocity vector
        dx = puck_current_pos[0] - puck_prev_pos[0]
        dy = puck_current_pos[1] - puck_prev_pos[1]
        speed_scale = 0.2  # Scale factor for velocity vector display
        plt.arrow(puck_current_pos[0], puck_current_pos[1], 
                  dx * speed_scale, dy * speed_scale, 
                  head_width=20, head_length=20, fc='blue', ec='blue', alpha=0.7)
        
        # Draw puck (previous and current positions)
        plt.plot(puck_prev_pos[0], puck_prev_pos[1], 'bo', markersize=10, label='Previous puck position')
        plt.plot(puck_current_pos[0], puck_current_pos[1], 'go', markersize=10, label='Current puck position')
        
        # Draw arm (current and target positions)
        plt.plot(arm_pos[0], arm_pos[1], 'ro', markersize=10, label='Current arm position')
        plt.plot(target_pos[0], target_pos[1], 'mo', markersize=10, label='Target arm position')
        
        # Connect current and target arm positions with a line
        plt.plot([arm_pos[0], target_pos[0]], [arm_pos[1], target_pos[1]], 'k-', alpha=0.5)
        
        # Add explanatory texts
        plt.text(puck_current_pos[0]+30, puck_current_pos[1], f'Puck ({puck_current_pos[0]:.0f}, {puck_current_pos[1]:.0f})', fontsize=9)
        plt.text(arm_pos[0]+30, arm_pos[1], f'Arm ({arm_pos[0]:.0f}, {arm_pos[1]:.0f})', fontsize=9)
        plt.text(target_pos[0]+30, target_pos[1], f'Target ({target_pos[0]:.0f}, {target_pos[1]:.0f})', fontsize=9)
        
        # Configure plot
        plt.xlim(-100, self.table_width + 100)
        plt.ylim(-100, self.table_height + 100)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.title('Air Hockey Scenario Visualization')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        
        return plt.gcf()  # Return created figure
     

def test_defender():
    """
    Test defense system with various scenarios and visualize results
    """
    # Table and arm parameters
    TABLE_WIDTH = 1600  # mm
    TABLE_HEIGHT = 800  # mm
    ARM_SPEED = 1500    # mm/s
    DEFENSE_LINE = 450  # mm
    
    defender = AirHockeyDefender(TABLE_WIDTH, TABLE_HEIGHT, ARM_SPEED, DEFENSE_LINE)
    
    test_scenarios = [
        {
            "name": "Straight movement to goal",
            "puck_prev": (800, 400),
            "puck_current": (750, 400),
            "arm_pos": (200, 300)
        },
        {
            "name": "Movement at an angle",
            "puck_prev": (1000, 300),
            "puck_current": (950, 350),
            "arm_pos": (200, 200)
        },
        {
            "name": "Arm far from interception point",
            "puck_prev": (800, 700),
            "puck_current": (750, 650),
            "arm_pos": (200, 100)
        },
        {
            "name": "Puck moving away from goal",
            "puck_prev": (300, 400),
            "puck_current": (350, 400),
            "arm_pos": (200, 400)
        },
        {
            "name": "Real example",
            "puck_prev": (623, 258),
            "puck_current": (577, 232),
            "arm_pos": (136, 252)
        }
    ]
    
    dt = 1/30  # 30 FPS camera
    
    for scenario in test_scenarios:
        print(f"\n=== Scenario: {scenario['name']} ===")
        puck_prev = scenario["puck_prev"]
        puck_current = scenario["puck_current"]
        arm_pos = scenario["arm_pos"]
        
        # Calculate puck velocity vector (for trajectory prediction)
        velocity, speed = defender.calculate_puck_vector(puck_prev, puck_current, dt)
        
        # Predict trajectory for visualization
        trajectory = defender.predict_trajectory(puck_current, velocity)
        
        # Get target position
        target_pos = defender.calculate_optimal_position(puck_prev, puck_current, arm_pos, dt)
        
        # Print information
        print(f"Previous puck position: {puck_prev}")
        print(f"Current puck position: {puck_current}")
        print(f"Current arm position: {arm_pos}")
        print(f"Target arm position: {target_pos}")
        
        # Create visualization
        defender.visualize_scenario(puck_prev, puck_current, arm_pos, target_pos, trajectory)
        
        # Save or show plot
        plt.savefig(f"scenario_{scenario['name'].replace(' ', '_')}.png", dpi=100, bbox_inches='tight')
        plt.close()  # Close current plot before creating the next one
        
    print("\nVisualizations saved as PNG files")


if __name__ == "__main__":
    test_defender()