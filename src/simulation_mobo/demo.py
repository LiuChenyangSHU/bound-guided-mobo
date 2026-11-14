"""An analytic two-objective example on the unit square."""


def evaluate(parameters):
    x, y = parameters["x"], parameters["y"]
    return {"quality_a": -(x * x + y * y) / 2,
            "quality_b": -((x - 1) ** 2 + (y - 1) ** 2) / 2}
