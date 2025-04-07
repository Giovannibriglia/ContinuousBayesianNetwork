from cbn.parameter_learning.brute_force import BruteForce
from cbn.parameter_learning.gp_pytorch import GP_gpytorch

ESTIMATORS = {"mle": BruteForce, "gp_gpytorch": GP_gpytorch}
