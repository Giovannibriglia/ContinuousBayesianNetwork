# Continuous and Scalable Bayesian Networks

A PyTorch-based Bayesian Network framework designed to handle **continuous** data, enabling continuous inference with **continuous CPDs**. It leverages GPU acceleration and batch operations, and is intended to support **dynamic structures** and **multi-agent** scenarios.


## Core Components

### 1. `BayesianNetwork`
- **Methods**:
  - `set_data`: Initialize or replace the dataset.
  - `add_data`: Extend the existing dataset (for dynamic BN setups).
  - `get_pdf(query)`: Retrieve the probability density function for a given query.
  - `get_cpd(query)`: Retrieve the conditional probability distribution for a query.
  - `plot_pdf(pdf, query)`: Plot the queried PDF.
  - `plot_cpd(cpd, query)`: Plot the queried CPD.
  - `print_structure()`: Visualize or print the network’s structure.

### 2. `Node`
- **Definition**: A node in the Bayesian Network, specified by its estimator, CPD estimator, and output distribution.
- **Methods**:
  - `get_pdf(query)`: Return the node’s PDF given a query.
  - `get_cpd(query)`: Return the node’s CPD given a query.

#### 2.1 Base Estimator (e.g., MLE, Bayesian, MCMC, ...)
- **Description**: Used to handle data fitting and parameter learning.
- **Method**:
  - `return_data(node_data, query)`: Process node-specific data in the context of a given query.

#### 2.2 Base CPD Estimator (Parametric or Non-Parametric)
- **Description**: Takes the data selected by the node’s base estimator and computes CPDs.
- **Method**:
  - `return_prob(selected_data)`: Return conditional probabilities for the selected data subset (parametric or non-parametric approach).

---

## Tasks

1. **Structure Learning**
   - *Planned Feature*: Automatic discovery of BN structure from data.

2. **Parameter Learning**
   - **MLE (Frequentist)**: Estimates parameters by maximizing the likelihood.
   - **Bayesian Estimation**: Incorporates priors and updates beliefs with observed data.

3. **Inference**
   - **Exact Inference**
     - Belief Propagation (message passing)
     - Junction Tree
     - Variable Elimination
   - **Causal Inference**
     - Backdoor adjustment
     - Frontdoor adjustment
     - Average Treatment Effect (ATE)
   - **Approximate Inference**
     - Markov Chain Monte Carlo (MCMC)
     - Variational Inference
     - Gibbs Sampling
     - Forward Sampling

4. **Representation**
   - *Planned Feature*: Methods to represent or export the BN structure and parameters.

---

## Pending Issues

1. **Batch Operation**
   - Ensuring consistent tensor shapes across multiple queries.
2. **Multi-Agent Interactions**
   - Handling data and structure for multiple interacting agents.
3. **High-Dimensional Data**
   - Efficient management and inference for large-scale continuous variables.

---


## License

[GNU GENERAL PUBLIC LICENSE](LICENSE)
