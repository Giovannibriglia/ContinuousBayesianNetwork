## Continuous and Scalable Bayesian Networks

**Goal:** a Bayesian Network designed to handle also continuous data, enabling continuous causal inference through continuous CPDs, fully implemented in PyTorch to leverage GPU acceleration and batch operations. It will support dynamic structures and multi-agent scenarios.


- Base classes:
	1) Bayesian Network_
		- set_data
		- add_data -> TODO: think about dynamic bn
		- get_pdf(query)
		- get_cpd(query)
		- plot_pdf(pdf, query)
		- plot_cpd(cpd, query)
		- print_structure()

	2) Node: defined with estimator, cpd_estimator, output_distribution
		- get_pdf(query)
		- get_cpd(query)

		1) Base Estimator (mle, be, mcmc,...)
			- return_data(node_data, query)

		2) Base CPD Estimator (parametric(distr) or non parametric(kde, gaussian processes))
			- return_prob(selected_data)

- Tasks:
	1) Structure Learning:
		TODO
	2) Parameter Learning:
		- MLE (frequentistic view)
		- Bayesian Estimator (bayesian view: data + prior)

	3) Inference:
		- Exact inference:
			- belief propagation (message passing)
			- junction tree
			- variable elimination
		- Causal Inference:
			- backdoor
			- frontdoor
			- average treatment effect
		- Approximate Inference:
			- Markov Chain Monte Carlo
			- Variational Inference
			- Gibbs Sampling
			- Forward Sampling
