# Efficient and Scalable Bayesian Networks

A **PyTorch-based Bayesian Network framework** that supports both **discrete** and **continuous** data types. Optimized
for GPU acceleration and batch operations, this framework is designed for dynamic network structures and multi-agent
scenarios, offering a flexible and scalable approach to learn and inference probabilities in complex domains.

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
| Status | Inference Kind | Engine                   | Auxiliary Library | Summary                                                            | Inference Complexity |
|--------|----------------|--------------------------|-------------------|--------------------------------------------------------------------|----------------------|
| Done   | Exact          | Variable Elimination     | PyTorch           | Eliminates variables to compute exact marginals                    | O(N × exp(W))        |
| TODO   | Exact          | Belief Propagation       | PyTorch           | Message passing in tree or loopy graphs for marginals              | O(N × d²)            |
| TODO   | Exact          | Junction Tree Algorithm  | PyTorch           | Converts graph into tree of cliques for efficient exact inference  | O(N × exp(T))        |
| TODO   | Approximate    | Markov Chain Monte Carlo | PyTorch           | Samples from posterior via random walk (e.g., Metropolis-Hastings) | O(K × N)             |
| TODO   | Approximate    | Variational Inference    | PyTorch           | Optimizes a simpler distribution to approximate the posterior      | O(I × N × D)         |
| TODO   | Approximate    | Gibbs Sampling           | PyTorch           | Iteratively samples each variable conditioned on the rest          | O(K × N × d)         |
| TODO   | Approximate    | Forward Sampling         | PyTorch           | Samples from joint by following the topological order              | O(N)                 |
| TODO   | Causal         | Backdoor Adjustment      | PyTorch           | Adjusts for confounders to estimate causal effects                 | O(N × d)             |
| TODO   | Causal         | Frontdoor Adjustment     | PyTorch           | Adjusts for mediators in specific structures                       | O(N × d²)            |
| TODO   | Causal         | Average Treatment Effect | PyTorch           | Computes expected difference in outcome from treatment vs. control | O(N)                 |

where:
- *N* is the number of variables
- *d* is the domain size per variable
- *W* is the tree-width of the graph (for VE and Junction Tree)
- *T* is the size of the largest clique in junction tree
- *K* is the number of MCMC samples
- *I* is the number of optimization iterations
- *D* is the dimensionality of approximate distribution

### Parameter Learning

| Status | Estimator              | Auxiliary Library | Summary                                                    | Learns Over Time      | Transferability                  | Training Complexity | Inference Complexity |
|--------|------------------------|-------------------|------------------------------------------------------------|-----------------------|----------------------------------|---------------------|----------------------|
| Done   | Brute Force Discrete   | PyTorch           | Stores and queries parameters from a discrete database     | Yes (adds data)       | Low – specific to data points    | O(N)                | O(1)                 |
| TODO   | Brute Force Continuous | PyTorch           | Explores full parameter space for continuous distributions | Yes (adds data)       | Low – specific to data points    | O(N × D)            | O(1)                 |
| TODO   | Bayesian Estimator     | PyTorch           | Incorporates priors and updates posterior over time        | Yes (Bayesian update) | High – priors and uncertainty    | O(N × D²)           | O(D²)                |
| Done   | Gaussian Processes     | GPyTorch          | Non-parametric model with uncertainty estimation           | Yes (online updates)  | Medium – kernel & prior settings | O(N³)               | O(N)                 |
| Done   | Linear Regression      | PyTorch           | Models linear relationship between inputs and output       | No                    | Medium – learned coefficients    | O(N × D²)           | O(D)                 |
| Done   | Logistic Regression    | PyTorch           | Classifier for binary or multiclass outputs                | No                    | Medium – learned coefficients    | O(N × D)            | O(D)                 |
| Done   | Neural Network         | PyTorch           | Learns non-linear functions from data                      | Yes (backpropagation) | High – feature representations   | O(E × N × D)        | O(D)                 |

where:
- *N* is the number of data points (samples)
- *D* is the number of features (dimensions)
- *E* is number of training epochs


### Input Data Support

- [X] Single Data Values
- [ ] Time Series Data
- [ ] Images
- [ ] Text Data

## Benchmarking

- [X] Gymnasium Environments for testing and benchmarking.
- [X] [Database with cause-effect pairs](https://webdav.tuebingen.mpg.de/cause-effect/)

## Pending Issues and Future Work

- **Multi-Agent Interactions:**
  Explore methods to combine Conditional Bayesian Networks (CBNs) to simulate and manage inter-agent interactions.
- **Adding knowledge over time**:
  - adding data
  - dynamic data
- **Current Problems:**
  - how do parameters/information change over time if also DAG changes?
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
