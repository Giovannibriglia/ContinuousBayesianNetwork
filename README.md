# Continuous and Scalable Bayesian Networks

A PyTorch-based Bayesian Network framework designed to handle **continuous** data, enabling continuous inference with **continuous CPDs**. It leverages GPU acceleration and batch operations, and is intended to support **dynamic structures** and **multi-agent** scenarios.


## Core Components

### 1. `BayesianNetwork`
- **Structure**:
  - Bayesian Network
    - Parameter Estimator
    - Probability Estimator
    - Inference

## Tasks

1. **Structure Learning**
   - *Planned Feature*: Automatic discovery of BN (dag) structure from data.

2. **Parameter Learning**
   - [X] **MLE**: Estimates parameters by maximizing the likelihood.
   - [ ] **Bayesian Estimation**: Incorporates priors and updates beliefs with observed data.

3. **Inference**
   - **Exact Inference**
     - [ ] Belief Propagation (message passing)
     - [ ] Junction Tree
     - [X] Variable Elimination
   - **Causal Inference**
     - [ ] Backdoor adjustment
     - [ ] Frontdoor adjustment
     - [ ] Average Treatment Effect (ATE)
   - **Approximate Inference**
     - [ ] Markov Chain Monte Carlo (MCMC)
     - [ ] Variational Inference
     - [ ] Gibbs Sampling
     - [ ] Forward Sampling

4. **Representation**
   - *Planned Feature*: Methods to represent or export the BN structure and parameters.

---

## Pending Issues

1. **Multi-Agent Interactions**
   - Handling data and structure for multiple interacting agents.
2. **High-Dimensional Data**
   - Efficient management and inference for large-scale continuous variables. For now, data are stored on CPU and, once filtered, are moved to GPU for core-computations.

---

## Current problems:
1. If the required configuration (even if is one of N queries) has no evidence on the acquired data, error.

## License

[GNU GENERAL PUBLIC LICENSE](LICENSE)
