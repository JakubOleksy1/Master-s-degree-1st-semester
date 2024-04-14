from figures import Rectangle, Square, Cube, Cuboid

cuboid = Cuboid(Square(3), 4)
print(cuboid.volume())
cube = Cube(Square(3))
print(cube.volume())