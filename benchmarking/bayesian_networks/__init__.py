from benchmarking.bayesian_networks.my_bn import MyCBN
from benchmarking.bayesian_networks.pgmpy_bn import PgmpyBN
from benchmarking.bayesian_networks.pyagrum import PyAgrumBN

BN_LIBRARIES = {"my_bn": MyCBN, "pgmpy": PgmpyBN, "pyagrum": PyAgrumBN}
