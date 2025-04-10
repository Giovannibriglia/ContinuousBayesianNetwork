from cbn.parameter_learning.brute_force import BruteForce
from cbn.parameter_learning.gp_pytorch import GP_gpytorch
from cbn.parameter_learning.linear_regression import LinearRegression

ESTIMATORS = {
    "brute_force": BruteForce,
    "gp_gpytorch": GP_gpytorch,
    "linear_regression": LinearRegression,
}
