class Rectangle():
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)

class Cube():
    def __init__(self, square: Square):
        self.base = square
        self.height = square.height
    def volume(self):
        return self.base.area() * self.height

    def surface_area(self):
        return self.base.area() * 6
    
class Cuboid():
    def __init__(self, figure, height):
        self.base = figure
        self.height = height

    def volume(self):
        return self.base.area() * self.height
    
    def surface_area(self):
        return self.base.area() * 2 + self.base.width * self.height * 2 + self.base.height * self.height * 2
