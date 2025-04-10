# Efficient and Scalable Bayesian Networks

A **PyTorch-based Bayesian Network framework** that supports both **discrete** and **continuous** data types. Optimized
for GPU acceleration and batch operations, this framework is designed for dynamic network structures and multi-agent
scenarios, offering a flexible and scalable approach to Bayesian inference.

## Overview

This project provides a modular framework with separate components for:
- **Structure Learning:** (Planned) Automatic discovery of network (DAG) structures from data.
- **Parameter Learning:** Various techniques from traditional maximum likelihood estimation to modern machine learning approaches.
- **Inference:** Methods for both exact and approximate inference, including causal analysis.

## Features

- **Modular Architecture:** Easily extendable components for structure, learning, and inference.
- **GPU Acceleration:** Utilizes PyTorch for efficient batch processing and scalable computation.
- **Dynamic & Multi-Agent Capable:** Designed to support evolving network structures and interactions between multiple Bayesian networks (CBNs).
- **Data Flexibility:** Handles single values, with future support planned for time series, images, and text.

## Core Components

### Structure
- **Bayesian Network Core**
  - **Nodes**:
  - **Parameter Learning**:
  - **Structure Learning**: *planned*
- **Inference**

### Inference Engine
- **Exact Inference:**
  - [ ] **Variable Elimination**
  - [ ] Belief Propagation (message passing)
  - [ ] Junction Tree Algorithm
- **Approximate Inference:**
  - [ ] Markov Chain Monte Carlo (MCMC)
  - [ ] Variational Inference
  - [ ] Gibbs Sampling
  - [ ] Forward Sampling
- **Causal Inference:**
  - [ ] Backdoor Adjustment
  - [ ] Frontdoor Adjustment
  - [ ] Average Treatment Effect (ATE)

### Parameter Learning Tasks

- [X] **Brute Force discrete MLE**: implements exhaustive likelihood maximization (acts as a database for parameter estimates).
- [ ] **Brute Force continuous MLE**: implements exhaustive likelihood maximization (acts as a database for parameter estimates).
- [ ] **Bayesian Estimation**: incorporates priors to update beliefs based on observed data.
- [X] **Gaussian Process #1**: integrates Gaussian processes for non-linear parameter estimation. Library: *gpytorch*
- [X] **Linear Regression**:
- [X] **Logistic Regression**:
- [X] **Deep Learning**:

### Input Data Support

- [X] Single Data Values
- [ ] Time Series Data
- [ ] Images
- [ ] Text Data

## Benchmarking

- [X] Gymnasium Environments for testing and benchmarking.
- [ ] VMAS Environments

## Pending Issues and Future Work

- **Multi-Agent Interactions:**
  Explore methods to combine Conditional Bayesian Networks (CBNs) to simulate and manage inter-agent interactions.
- **Current Problems:**
  - [List any current bugs or issues here]
- **Additional Enhancements:**
  - Further methods for both parameter learning and inference are being researched and implemented.

## Contributing

Contributions, issues, and feature requests are welcome! Please check the [issues page](./issues) or submit a pull request if you have suggestions, bug fixes, or improvements.

## License

Distributed under the [GNU GENERAL PUBLIC LICENSE](LICENSE).

## Citation

If you use this framework in your research or projects, please cite it as follows:
```

```
