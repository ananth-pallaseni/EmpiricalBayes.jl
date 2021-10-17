# EmpiricalBayes

[![Build Status](https://travis-ci.org/ananth-pallaseni/EmpiricalBayes.jl.svg?branch=master)](https://travis-ci.org/ananth-pallaseni/EmpiricalBayes.jl)
[![codecov.io](http://codecov.io/github/ananth-pallaseni/EmpiricalBayes.jl/coverage.svg?branch=master)](http://codecov.io/github/ananth-pallaseni/EmpiricalBayes.jl?branch=master)

## Description
EmpiricalBayes is a package for applying prior information to a set of test statistics using the method of empirical Bayes [[1]](https://en.wikipedia.org/wiki/Empirical_Bayes_method) .

As it stands the package contains basic functions to calculate empirical Bayes posteriors from lists of test statistics and prior values. This package is intended to be used as part of other workflows, where it can be extended with glue functions that fit the particular data structures in use. A typical use case would be:
- In SomeOtherPackage.jl: implement a function that breaks down the native data structures or formats into the matched lists required by `empirical_bayes`.
- Use `empirical_bayes` to calculate posteriors.
- Convert the posterior list back into the native structures and continue the workflow.

See [NetworkInference.jl](https://github.com/Tchanders/NetworkInference.jl) for an example use case (and how to optionally load the glue functions only when EmpiricalBayes is present).


## Installation
Use ']' to enter Pkg mode, then
```julia
pkg> add https://github.com/ananth-pallaseni/EmpiricalBayes.jl
```

## Basic Usage
Use the ```empirical_bayes``` function as follows:

```julia
empirical_bayes(test_statistics, priors, num_bins, distr)
```

where `test_statistics` and `priors` are lists such that `priors[i]` is the prior for `test_statistics[i]`, `num_bins` is the number of bins to discretize into and `distr` is the form of the null distribution (currently `Gamma` and `Normal` are supported - alternatively `:Gamma` and `:Normal`, if the caller is not using Distributions.jl).

This returns a list of posterior values such that `output[i]` is the posterior corresponding to applying `priors[i]` to `test_statistic[i]`.

## API
EmpiricalBayes exposes five major functions:

```julia
"""
Bin the values in test_statistics into n uniform width bins.
Returns the midpoints of the bins, the counts in each bin and the bin width.
"""
discretize_test_statistics(test_statistics::AbstractArray{<:Real}, n)
```

```julia
"""
Fit a distribution to the discretized inputs. Uses mode-matching, since
the input test statistics may have been truncated.

Returns the fitted distribution

proportion_to_keep is the proportion of lowest valued test_statistics to keep before
applying the mode-matching algorithm.

verbose is whether or not to print the proportion of values kept.
"""
fit_null_distribution(midpoints, counts, num_bins, bin_width, proportion_to_keep, distr, verbose=true)
```

```julia
"""
Fit a cubic spline distribution to the discretized inputs.
Returns a function that maps from x->pdf of mixture at x
"""
fit_mixture_distribution(midpoints, counts, bin_width)
```

```julia
"""
Calculate the empirical Bayes posterior using the priors, null distribution and
mixture distribution.

Returns a list of posterior values such that `output[i]` is the posterior corresponding
to applying `priors[i]` to `test_statistic[i]`.

w0 is the base value for prior calculation - will probably never need to be changed.

tail is whether to treat the test as two-tailed (:two) or one-tailed (:lower or :upper)
"""
calculate_posterior(priors, test_statistics, null_distr, mixture_pdf, tail, w0=0.0)
```

```julia
"""
Calculate the empirical Bayes posteriors of the input statistics using the priors.
Shorthand for calling the above functions in order.
Returns a list of posterior values such that `output[i]` is the posterior corresponding
to applying `priors[i]` to `test_statistic[i]`.

proportion_to_keep is the proportion of lowest valued test_statistics to keep before
applying the mode-matching algorithm when fitting the null distribution.
"""
empirical_bayes(test_statistics, priors, num_bins, distr, proportion_to_keep=1.0, tail=:two, w0=0.0)
```
