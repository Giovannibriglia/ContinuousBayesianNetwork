from typing import Dict

import torch

from cbn.base.bayesian_network import BayesianNetwork


def get_distribution_parameters(dist: torch.distributions.Distribution):
    """
    Retrieve the parameters of a PyTorch distribution.

    Args:
        dist (torch.distributions.Distribution): A PyTorch distribution object.

    Returns:
        dict: A dictionary of distribution parameters.
    """
    return {key: getattr(dist, key) for key in vars(dist) if not key.startswith("_")}


def define_cbn(data, dag, print_structure: bool = False):
    bn = BayesianNetwork(dag=dag, data=data)

    if print_structure:
        bn.print_bn_structure()

    return bn


def get_cpd_and_pdf(
    cbn_obj: BayesianNetwork,
    target_feature: str,
    evidence: Dict,
    uncertainty: float = 0.1,
    normalize_pdf: bool = False,
):

    cpd, pdf, target_values = cbn_obj.get_cpd_and_pdf(
        target_feature, evidence, uncertainty
    )

    if normalize_pdf:
        pdf = pdf_normalization(pdf)

    return cpd, pdf, target_values


def pdf_normalization(pdf: torch.Tensor):
    min_val = torch.min(pdf)
    max_val = torch.max(pdf)
    return (pdf - min_val) / (max_val - min_val)


def get_pdf_value(value: float, target_values: torch.Tensor, pdf: torch.Tensor):
    """
    Given a target value, find its closest match in target_values and return the corresponding PDF value.

    Args:
        value (float): The value at which the PDF is required.
        target_values (torch.Tensor): Tensor of shape (N,) representing points where the PDF is evaluated.
        pdf (torch.Tensor): Tensor of shape (N,) representing the PDF values at target_values.

    Returns:
        float: The PDF value at the closest target point.
    """
    index = torch.argmin(torch.abs(target_values - value))  # Find the closest index
    return pdf[index].item()  # Return the corresponding PDF value
