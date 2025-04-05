from cbn.parameter_learning.gp_pytorch import GP_gpytorch
from cbn.parameter_learning.mle import MLE

ESTIMATORS = {"mle": MLE, "gp_gpytorch": GP_gpytorch}
