"""
Functions for applying empirical Bayes to a set of test statistics.
""";

using Discretizers
using Distributions
using Interpolations
using DataFrames
using GLM


"""
    discretize_test_statistics(test_statistics::AbstractArray{<:AbstractFloat, 1}, n::Int)

Bin the values in test_statistics into n uniform width bins.
Returns the midpoints of the bins, the counts in each bin and the bin width.
"""
function discretize_test_statistics(test_statistics::AbstractArray{<:AbstractFloat, 1}, n::Int)
    # Calculate midpoints
    min_ts, max_ts = extrema(test_statistics)
    bin_width = (max_ts - min_ts) / n
    midpoints = [min_ts + i*bin_width - bin_width/2 for i in 1:n]

    # Get counts for each bin
    bin_edges = binedges(DiscretizeUniformWidth(n), test_statistics)
    lindisc = LinearDiscretizer(bin_edges)
    counts = get_discretization_counts(lindisc, test_statistics)

    return midpoints, counts, bin_width
end


"""
    keep_bottom_proportion_of_hist(midpoints, counts, proportion_to_keep)

Remove the highest proportion of values from a histogram. Returns the new
midpoints and counts for the truncated histogram.

# Arguments
- `midpoints` : list of midpoints of the histogram bins.
- `counts` : list of counts for the histogram bins.
- `proportion_to_keep` : proportion of lowest values in the histogram to keep.
"""
function keep_bottom_proportion_of_hist(midpoints, counts, proportion_to_keep)
    # Number of values to keep
    num_vals_total = sum(counts)
    to_keep = Int(round(proportion_to_keep * num_vals_total))

    # Iterate through counts to find which bins to keep
    # Keep updating total until it is more than we want to keep
    cur_bin = 1
    cur_total = 0
    while (cur_total < to_keep) && (cur_bin <= length(counts))
        cur_total += counts[cur_bin]
        cur_bin += 1
    end

    # Remove bins containing values that are too high
    counts = counts[1:cur_bin-1]
    midpoints = midpoints[1:cur_bin-1]

    # Remove excess values from the last bin that stays
    counts[cur_bin-1] -= cur_total - to_keep

    return midpoints, counts
end

"""
    estimate_poisson_parameters(num_bins, bin_width, midpoints, counts)

Estimates the poisson parameters required to fit a mode matched null distribution
to the input histogram. Solves the regression problem outlined in equations (6)
and (8) in Schwarzman 2009 (https://projecteuclid.org/euclid.aoas/1231424213).
Returns the parameters C, eta1, eta2
"""
function estimate_poisson_parameters(num_bins, bin_width, midpoints, counts)
    # Get the log of the counts
    log_counts = log.(counts)

    # Filter out non finite
    log_counts = [isfinite(x) ? x : eps() for x in log_counts]

    # Minus difference h to create regression target
    h = log(num_bins * bin_width)
    y = log_counts - h


    # Construct and solve the regression problem
    data = DataFrame(Y = y, X2 = midpoints, X3 = log.(midpoints))
    C, eta1, eta2 = coef(glm(@formula(Y ~ X2 + X3), data, Normal(), minStepFac = eps()))

    return C, eta1, eta2
end



"""
    get_gamma_parameters(C, eta1, eta2)

Gets the parameters which describe a gamma distribution based on the poisson
parameters input. Converts the poisson parameter inputs into a chi squared
distribution, then converts that into a gamma distribution.
"""
function get_gamma_parameters(C, eta1, eta2)

    # Convert eta to chi squared parameters
    a = -(1 / (2*eta1))
    v = 2 * (eta2 + 1)

    # Convert chi squared parameters to gamma shape and scale parameters
    k = v/2
    theta = 2*a

    # Find p0 by solving the following:
    # log(p0) = C + log(gammafunc(eta2 + 1) / (-eta1)^(eta2 + 1))
    # p0 = exp(C + log(gamma(eta2 + 1) / (-eta1)^(eta2 + 1)))
    p0 = 0


    return k, theta, p0
end

"""
    fit_null_distribution(midpoints, counts, num_bins, bin_width, proportion_to_keep)

Fit a gamma distribution to the discretized inputs. Usues the mode matching
method outlined in Schwarzman 2009: https://projecteuclid.org/euclid.aoas/1231424213.
"""
function fit_null_distribution(midpoints, counts, num_bins, bin_width, proportion_to_keep)
    # Remove highest test statistics, that dont fit the zero assumption
    midpoints, counts = keep_bottom_proportion_of_hist(midpoints, counts, proportion_to_keep)

    # Estimate poisson parameters using regression
    C, eta1, eta2 = estimate_poisson_parameters(num_bins, bin_width, midpoints, counts)

    # Calculate gamma parameters
    k, theta, p0 = get_gamma_parameters(C, eta1, eta2)

    params_valid = k > zero(k) && theta > zero(theta)
    if !params_valid
        error_string = "Error: resulting gamma parameters are invalid. Got k = ", k, " and Θ = ", theta, ".
        Input data might not be gamma distributed, or number of bins needs to change.
        Exiting"
        error(error_string)
    end

    # Create gamma distribution
    f0_distr = Gamma(k, theta)
    f0(x) = pdf(f0_distr, x)

    return f0
end

"""
    fit_mixture_distribution(midpoints, counts, bin_width)

Fit a cubic spline distribution to the discretized inputs.
"""
function fit_mixture_distribution(midpoints, counts, bin_width)
    # Add some zero bins on the beginning to prevent curve going negative
    number_of_zero_bins = 10 #3
    zero_bins = zeros(number_of_zero_bins)
    unshift!(counts, zero_bins...)
    for i in 1 : number_of_zero_bins
        unshift!(midpoints, midpoints[1] - bin_width)
    end

    # Normalize counts
    num_statsitics = sum(counts)
    counts = counts / (num_statsitics * bin_width)

    # Fit spline to histogram points
    interpolation = interpolate(counts, BSpline(Cubic(Line())), OnCell())
    r = midpoints[1] : (midpoints[end]-midpoints[1]) / (length(midpoints)-1) : midpoints[end]
    scaled_interpolation = Interpolations.scale(interpolation, r)

    fhat(x) = max(scaled_interpolation[x], 0)
    fhat(x::AbstractArray) = [fhat(i) for i in x]

    return fhat
end


"""
    calculate_posterior(priors, null_distr, mixture_distr)

Calculate the empirical Bayes posterior using the priors, null distribution and
mixture distribution.
"""
function calculate_posterior(priors, null_distr, mixture_distr)
  nothing
end
