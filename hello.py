# TODO read the miOSQP paper
# http://cse.lab.imtlucca.it/~bemporad/publications/papers/ecc18-miosqp.pdf
#
# TODO read about Pajarito, a mixed-integer convex program solver in Julia.
# https://github.com/jump-dev/Pajarito.jl
# Research team is from MIT.
# Pajarito implements an algorithm that uses a continuous conic solver
# and a MILP solver; it uses the continuous relaxation to add cuts
# to the MILP. Pajarito relies on external conic and MILP solvers.
# It can use HiGHS as the MILP solver.

# MIQP example from MATLAB: iteratively adding constraints to a
# MILP to approximate the quadratic objective.
# https://www.mathworks.com/help/optim/ug/mixed-integer-quadratic-programming-portfolio-optimization-solver-based.html

import pulp
import numpy as np


def calc_slope_intercept_from_points(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return (slope, intercept)


def create_quadratic_approx(a: pulp.LpAffineExpression, da: float = 0.1, nsegs: int = 10, name: str = "quadratic approx") -> tuple[pulp.LpVariable, list[pulp.LpConstraint]]:
    c = pulp.LpVariable(name, lowBound=0.0)
    

# Solve a simple linear regression problem, where the intercept must be an integer.

prob = pulp.LpProblem()
slope = pulp.LpVariable("slope")
intercept = pulp.LpVariable("intercept", -100, 100, pulp.LpInteger)
a = pulp.LpAffineExpression([(slope, 2.0), (intercept, 1.0)])
b = 2.0 * a + 3.0
print(b)

