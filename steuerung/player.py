class Player:
    def __init__(self, position):
        self.position = position
        self.state = "Idle"

    def update(self, puck_position):
        self.transition(puck_position)
        self.act(puck_position)

    def transition(self, puck_position):
        distance_to_goal = puck_position.distance_to(self.goal_position)

        if distance_to_goal < 100:
            self.state = "Defend"
        elif puck_position.x > self.position.x:
            self.state = "Attack"
        else:
            self.state = "Idle"

    def act(self, puck_position):
        if self.state == "Idle":
            self.move_towards_center()
        elif self.state == "Defend":
            self.move_to_block(puck_position)
        elif self.state == "Attack":
            self.move_to_hit(puck_position)

    def move_towards_center(self):
        # Move toward center of your side
        pass

    def move_to_block(self, puck_position):
        # Try to position between puck and goal
        pass

    def move_to_hit(self, puck_position):
        # Move to hit the puck toward the opponent's goal
        pass