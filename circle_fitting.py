from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

from ransac import RANSAC

PI = np.pi

POINTS = np.array(
    [
        [0, 0],
        [0, -10],
        [10, 0],
        [20, 0],
        [3, -3],
        [1, -3],
        [1, 0.5],
        [2, 0.5],
        [3, 0.5],
        [4, -1],
        [0, -1],
        [0, -2],
        [5, 5],
    ]
)


@dataclass
class Point:
    x: float
    y: float

    @staticmethod
    def from_array(arr) -> "Point":
        return Point(arr[0], arr[1])

    @staticmethod
    def create_list(arr) -> List["Point"]:
        return [Point(x[0], x[1]) for x in arr]


@dataclass
class Circle:
    center: Point
    radius: float

    @property
    def x(self):
        return self.center.x

    @property
    def y(self):
        return self.center.y

    def get_point_at_angle(self, a):
        return Point(
            self.x + np.cos(a) * self.radius,
            self.y + np.sin(a) * self.radius,
        )

    def draw(self):
        angles = np.linspace(0, 2 * PI, 30)
        points = [self.get_point_at_angle(a) for a in angles]
        plt.axis("equal")
        plt.plot([p.x for p in points], [p.y for p in points], color="red")
        plt.scatter(self.x, self.y, marker="x", color="red")

    def __str__(self):
        return f"x={self.x:.2f}, y={self.y:.2f}, R={self.radius:.2f}"


def draw_points(points: List[Point], marker="o", color="blue"):
    plt.scatter(
        [p.x for p in points], [p.y for p in points], marker=marker, color=color
    )


def fit_circle(points: List[Point]) -> Optional[Circle]:
    a = [[-2 * p.x, -2 * p.y, 1] for p in points]
    a = np.array(a).reshape((-1, 3))

    b = [-p.x**2 - p.y**2 for p in points]
    b = np.array(b).reshape((-1, 1))

    try:
        solution = np.linalg.inv(a.T @ a) @ a.T @ b
    except np.linalg.LinAlgError as error:
        print("Cannot compute solution:", error)
        return None
    x, y, t = solution.flatten()

    r = np.sqrt(x**2 + y**2 - t)
    return Circle(Point(x, y), r)


def compute_error(p: Point, circle: Circle) -> float:
    # find coordinates of intersection between
    # the radius (passing through a given point) and the circle itself
    # i.e. find angle of radius corresponding to the point
    angle = np.arctan2(p.y - circle.y, p.x - circle.x)

    c = circle.get_point_at_angle(angle)

    return np.linalg.norm(np.array([p.x, p.y]) - np.array([c.x, c.y]))


def inlier_check(point, circle):
    return compute_error(point, circle) < 1


def main():
    points = Point.create_list(POINTS)

    ## Raw solution, messed up by outliers
    # circle = fit_circle(points)

    circle, inliers = RANSAC(fit_circle, inlier_check).run(50, points, 4, 0.65)

    print(circle)
    circle.draw()
    draw_points(points)
    draw_points(inliers, marker="+", color="yellow")
    plt.show()


if __name__ == "__main__":
    main()
