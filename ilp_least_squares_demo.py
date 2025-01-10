"""This script demonstrates adding a quadratic objective to a mixed integer linear program (MILP)
by approximating the quadratic objective with piecewise linear constraints.

This is a weird thing to do, and there may be better ways to solve a mixed-integer quadratic program (MIQP):

- Gurobi and CPLEX are commercial solvers that can solve MIQPs efficiently.

- SCIP is an open-source solver that can solve MIQPs.
    However, SCIP is much slower than HiGHS at solving large linear problems.

- [miOSQP](https://github.com/osqp/miosqp) is an open-source MIQP solver, but it has not been maintained since 2020.

- [Pajarito](https://github.com/jump-dev/Pajarito.jl) can solve mixed-integer convex programs
    by adding constraints to a MILP to approximate the convex objective. It's a more general and better
    version of the approach demonstrated in this script. It can use HiGHS as the MILP solver.
"""

import pulp
import numpy as np


def calc_slope_intercept_from_points(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """Calculate the slope and intercept of a line that passes through the points (x1, y1) and (x2, y2)."""
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return (slope, intercept)


def create_quadratic_approx(
    a: pulp.LpAffineExpression,
    da: float = 0.5,
    nsegs_per_side: int = 10,
    name: str = "quadratic approx",
) -> tuple[pulp.LpVariable, list[pulp.LpConstraint]]:
    """Create variable constrained to be greater than `a` squared, approximately.

    Args:
        `da`: The width of each segment in the piecewise linear approximation.
        `nsegs_per_side`: The number of segments on teach of the positive and negatives
            sides of the piecewise linear approximation.

    Returns:
        A new variable `c` and a list of `2 * nseg` constraints that enforce `c >= a**2`
        approximately, using a piecewise linear approximation.

    Note: if `nsegs_per_side = 1`, then the constraints exactly enforce `c >= abs(a)`.
    """
    c = pulp.LpVariable(name, lowBound=0.0)
    constraints = []
    for i in range(nsegs_per_side):
        x1 = i * da
        x2 = (i + 1) * da
        y1 = x1**2
        y2 = x2**2
        slope, intercept = calc_slope_intercept_from_points(x1, y1, x2, y2)
        constraints.append(c >= slope * a + intercept)
        slope, intercept = calc_slope_intercept_from_points(-x1, y1, -x2, y2)
        constraints.append(c >= slope * a + intercept)
    return c, constraints


# Solve a simple linear regression problem, where the intercept must be an integer.
SLOPE_TRUE = 1.87
INTERCEPT_TRUE = -2
# Generate random data points around the true line.
rng = np.random.default_rng(45132)
xs = np.linspace(-10, 10, 20)
ys = SLOPE_TRUE * xs + INTERCEPT_TRUE + rng.normal(0, 1.0, len(xs))

prob = pulp.LpProblem()
# The optimization variables are the slope and intercept of the fit line.
slope = pulp.LpVariable("slope")
intercept = pulp.LpVariable("intercept", -100, 100, pulp.LpInteger)

# For each data point, create an approximate of the squared error.
errors_squared_approx = []
quad_approx_constraints = []
for x, y in zip(xs, ys):
    c, constraints = create_quadratic_approx(
        y - slope * x - intercept, da=0.1, nsegs_per_side=20, name=f"quad approx at x={x:.3f}"
    )
    errors_squared_approx.append(c)
    quad_approx_constraints.append(constraints)

# Minimize the sum of the squared errors.
prob += pulp.lpSum(errors_squared_approx), "Sum of squared errors"
# Enforce the constraints of the quadratic approximations.
for x, constraints in zip(xs, quad_approx_constraints):
    for i, constraint in enumerate(constraints):
        prob += constraint, f"quad approx segment {i} at x={x:.3f}"

prob.solve(solver=pulp.HiGHS(timeLimit=30.0, gapRel=0.01))

print("Status:", pulp.LpStatus[prob.status])
print(f"slope: fit = {slope.varValue:.3f}, true = {SLOPE_TRUE:.3f}")
print(f"intercept: fit = {intercept.varValue:.0f}, true = {INTERCEPT_TRUE:d}")
errors = ys - slope.varValue * xs - intercept.varValue
print(
    f"Sum of squared errors: approx = {prob.objective.value():.3f}, true = {np.sum(errors**2):.3f}"
)
