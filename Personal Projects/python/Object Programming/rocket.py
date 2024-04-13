from random import randint

class Rocket:
    """A class to track rockets."""

    def __init__(self, speed = 1):
        """Initialize attributes for the rocket."""
        self.height = 0
        self.speed = speed

    def move_up(self):
        """Move the rocket up and increase its height."""
        self.height += 1 * self.speed

    def __str__(self):
        return f"Rocket at height {self.height} with speed {self.speed}"
    
class RocketBoard:
    def __init__(self, amountOfRockets = 5):
        self.rockets = [
            Rocket(randint(1, 6))
            for _ in range(amountOfRockets)
            ]

        for _ in range(10):
            rocketIndexToMove = randint(0, len(self.rockets) - 1)
            self.rockets[rocketIndexToMove].move_up()

        for rocket in self.rockets:
            print(rocket)  
