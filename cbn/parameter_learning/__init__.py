from cbn.parameter_learning.brute_force import BruteForce
from cbn.parameter_learning.gp_gpytorch import GP_gpytorch
from cbn.parameter_learning.linear_regression import LinearRegression
from cbn.parameter_learning.logistIc_regression import LogisticRegression
from cbn.parameter_learning.neural_network import NeuralNetwork

ESTIMATORS = {
    "brute_force": BruteForce,
    "gp_gpytorch": GP_gpytorch,
    "linear_regression": LinearRegression,
    "logistic_regression": LogisticRegression,
    "neural_network": NeuralNetwork,
}
