"""
Functions for applying empirical Bayes to a set of test statistics.
""";

using Discretizers
using Distributions
using Interpolations
using DataFrames
using GLM


"""
    discretize_test_statistics(test_statistics[, discretization])

Bin the values in test_statistics into n uniform width bins.
Returns the midpoints of the bins, the counts in each bin and the bin width.
"""
function discretize_test_statistics(test_statistics, n)
    # Calculate midpoints
    min_ts, max_ts = extrema(test_statistics)
    bin_width = (max_ts - min_ts) / n
    midpoints = [i*bin_width - bin_width/2 for i in 1:n]

    # Get counts for each bin
    bin_edges = binedges(DiscretizeUniformWidth(n), test_statistics)
    lindisc = LinearDiscretizer(bin_edges)
    counts = get_discretization_counts(lindisc, test_statistics)

    return midpoints, counts, bin_width
end


"""
    fit_null_distribution(midpoints, counts, num_bins, bin_width, proportion_to_keep)

Fit a gamma distribution to the discretized inputs. Usues the mode matching
method outlined in Schwarzman 2009: https://projecteuclid.org/euclid.aoas/1231424213.
"""
function fit_null_distribution(midpoints, counts, num_bins, bin_width, proportion_to_keep)
    nothing
end

"""
    fit_mixture_distribution(midpoints, counts, bin_width)

Fit a cubic spline distribution to the discretized inputs.
"""
function fit_mixture_distribution(midpoints, counts, bin_width)
  nothing
end


"""
    calculate_posterior(priors, null_distr, mixture_distr)

Calculate the empirical Bayes posterior using the priors, null distribution and
mixture distribution.
"""
function calculate_posterior(priors, null_distr, mixture_distr)
  nothing
end
