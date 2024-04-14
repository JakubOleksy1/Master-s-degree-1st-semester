from random import randint
from math import sqrt

class Rocket:
    """A class to track rockets."""

    nextId = 1

    def __init__(self, speed = 1, height = 0, length = 0, fuel = 100):
        """Initialize attributes for the rocket."""
        self.height = height
        self.length = 0
        self.fuel = 100
        self.speed = speed

        self.id = Rocket.nextId
        Rocket.nextId += 1

    def move_up(self):
        """Move the rocket up and increase its height."""
        self.height += 1 * self.speed
        self.fuel -= 1 * self.speed

    def move_sideways(self):
        if(randint(0, 1) == 0):
            self.length -= 1
        else:
            self.length += 1
        self.fuel -= 1        

    def __str__(self):
        return f"Rocket parameters:\nheight: {self.height}\nspeed: {self.speed}\nlength: {self.length}\nfuel: {self.fuel}\n"
    
class RocketBoard:
    def __init__(self, amountOfRockets = 5):
        self.rockets = [
            Rocket(randint(1, 3), 0, 0, 100)
            for _ in range(amountOfRockets)
            ]

        for _ in range(10):
            rocketIndexToMove = randint(0, len(self.rockets) - 1)
            self.rockets[rocketIndexToMove].move_up()
            self.rockets[rocketIndexToMove].move_sideways()

        for rocket in self.rockets:
            print(rocket)  

    def __getitem__(self, key):
        return self.rockets[key]

    def __setitem__(self, key, value):
        self.rockets[key].height = value

    @staticmethod
    def get_distance(rocket1: Rocket, rocket2: Rocket) -> float:
        distanceY = (rocket1.height - rocket2.height) ** 2
        distanceX = (rocket1.length - rocket2.length) ** 2
        return sqrt(distanceX + distanceY)

    def get_amout_of_rockets(self):
        return len(self.rockets)

    def __len__(self):
        return self.get_amout_of_rockets()
