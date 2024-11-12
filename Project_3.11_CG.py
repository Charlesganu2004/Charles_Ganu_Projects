

import matplotlib.pyplot as plt


class Polygon:
    def __init__(self, verts):
        self.verts = verts

    def plot(self, point=None):
        X = [v[0] for v in self.verts]
        Y = [v[1] for v in self.verts]
        plt.plot(X + [self.verts[0][0]], Y + [self.verts[0][1]], 'b-')
        if point:
            plt.plot(point[0], point[1], 'ro')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Polygon')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def inside(self, R):
        X, Y = R
        wind_numb = 0
        for i in range(len(self.verts)):
            Xi, Yi = self.verts[i]
            Xi1, Yi1 = self.verts[(i + 1) % len(self.verts)]
            if Yi <= Y:
                if Yi1 > Y:
                    if self.is_left(Xi, Yi, Xi1, Yi1, X, Y) > 0:
                        wind_numb += 1
            else:
                if Yi1 <= Y:
                    if self.is_left(Xi, Yi, Xi1, Yi1, X, Y) < 0:
                        wind_numb -= 1
        return wind_numb != 0

    def is_left(self, X0, Y0, X1, Y1, X, Y):
        return (X1 - X0) * (Y - Y0) - (X - X0) * (Y1 - Y0)


if __name__ == "__main__":
    verts1 = [(5, 1), (2, 3), (-2, 3.5), (-4, 1), (-2, 1.5), (-2, -2), (-5, -3), (2, -2.5), (5.5, -1)]
    plyg1 = Polygon(verts1)
    R1 = (0, 0)
    R2 = (-4, 0)
    print("Point R1 is inside polygon:", plyg1.inside(R1))
    print("Point R2 is inside polygon:", plyg1.inside(R2))
    plyg1.plot(R1)
    plyg1.plot(R2)

    verts2 = [(4, 1), (1, 2), (-1, 1), (-4, 2), (-5, -2), (-3, -2), (-5, -3), (2, -2), (5, -2)]
    plyg2 = Polygon(verts2)
    R3 = (0, 0)
    R4 = (-4, 0)
    print("Point R3 is inside polygon:", plyg2.inside(R3))
    print("Point R4 is inside polygon:", plyg2.inside(R4))
    plyg2.plot(R3)
    plyg2.plot(R4)
