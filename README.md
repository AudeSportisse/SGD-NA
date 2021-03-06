# README


This repository hosts the code to use stochastic gradient descent (SGD) or averaged stochastic gradient descent (AvSGD) in the linear model when the covariates may contain heterogeneous MCAR missing values.

There are the following Notebooks:

* for analysing the convergence behaviors of the SGD and AvSGD algorithms on synthetic data **debiasingSGDforNA_synthetic_data.ipynb**.

* for estimating the coefficient parameter in a real dataset (with real missing values): **debiasingSGDforNA_non_oracle.ipynb**.

* for introducing missing values in a real dataset and computing the prediction error considering that the test set is fully observed **debiasingSGDforNA_superconductivity_data.ipynb**.

* for running the SGD and AvSGD algorithms on the superconductivity dataset with polynomial features **superconductivity_poly_features.ipynb**. 
