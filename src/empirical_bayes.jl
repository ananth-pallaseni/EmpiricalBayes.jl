"""
Functions for applying empirical Bayes to a set of test statistics.
""";

using Discretizers
using Distributions
using Interpolations
using DataFrames
using GLM


"""
    discretize_test_statistics(test_statistics::AbstractArray{<:Real}, n)

Bin the values in test_statistics into n uniform width bins.
Returns the midpoints of the bins, the counts in each bin and the bin width.
"""
function discretize_test_statistics(test_statistics::AbstractArray{<:Real}, n)
    try
        n = Int(n) # Bad form, but allows values like 10.0 to be silently used as 10
    catch e
        error("n must be an Int or convertible to an Int")
    end

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
    keep_bottom_proportion_of_hist(midpoints, counts, proportion_to_keep, verbose)

Remove the highest proportion of values from a histogram. Returns the new
midpoints and counts for the truncated histogram.

# Arguments
- `midpoints` : list of midpoints of the histogram bins.
- `counts` : list of counts for the histogram bins.
- `proportion_to_keep` : proportion of lowest values in the histogram to keep.
"""
function keep_bottom_proportion_of_hist(midpoints, counts, proportion_to_keep, verbose)
    @assert 0.0 < proportion_to_keep <= 1.0

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

    # Only keep bins whose entire counts are within the kept interval
    final_bin = cur_total == to_keep ? cur_bin-1 : cur_bin-2
    counts = counts[1:final_bin]
    midpoints = midpoints[1:final_bin]

    if verbose
        kept_proportion = sum(counts) / num_vals_total
        println("Kept $kept_proportion of test statistics")
    end

    return midpoints, counts
end


"""
    estimate_fit_parameters(num_bins, bin_width, midpoints, counts, distr::Type{Gamma})

Estimates the poisson parameters required to fit a mode matched null distribution
to the input histogram. Solves the regression problem outlined in equations (6)
and (8) in Schwarzman 2009 (https://projecteuclid.org/euclid.aoas/1231424213).
Returns the parameters C, eta1, eta2
"""
function estimate_fit_parameters(num_bins, bin_width, midpoints, counts, distr::Type{Gamma})
    # Get the log of the counts
    log_counts = log.(counts)

    # Filter out non finite
    log_counts = [isfinite(x) ? x : eps() for x in log_counts]

    # Minus difference h to create regression target
    h = log(num_bins * bin_width)
    y = log_counts - h


    # Construct and solve the regression problem
    data = DataFrame(Y = y, X1 = midpoints, X2 = log.(midpoints))
    C, eta1, eta2 = coef(glm(@formula(Y ~ X1 + X2), data, Normal(), minStepFac = eps()))

    return C, eta1, eta2
end
"""
    estimate_fit_parameters(num_bins, bin_width, midpoints, counts, distr::Type{Normal})

Estimates the parameters required to fit a mode matched null distribution to the
input histogram. Solves the regression problem outlined in equation (4.5) and (4.6)
in Efron 2007 (https://projecteuclid.org/euclid.aos/1188405614).
Returns the parameters beta0, beta1, beta2
"""
function estimate_fit_parameters(num_bins, bin_width, midpoints, counts, distr::Type{Normal})
    # Get the log of the counts
    log_counts = log.(counts)

    # Filter out non finite
    log_counts = [isfinite(x) ? x : eps() for x in log_counts]

    # Construct and solve the regression problem
    data = DataFrame(Y = log_counts, X1 = midpoints, X2 = midpoints.^2)
    beta0, beta1, beta2 = coef(glm(@formula(Y ~ X1 + X2), data, Normal(), minStepFac = eps()))

    return beta0, beta1, beta2
end


"""
    get_distr_parameters(C, eta1, eta2, distr::Type{Gamma})

Gets the parameters which describe a gamma distribution based on the poisson
parameters input. Converts the poisson parameter inputs into a chi squared
distribution, then converts that into a gamma distribution.
"""
function get_distr_parameters(C, eta1, eta2, distr::Type{Gamma})

    # Convert eta to chi squared parameters
    a = -(1 / (2*eta1))
    v = 2 * (eta2 + 1)

    # Convert chi squared parameters to gamma shape and scale parameters
    k = v/2
    theta = 2*a

    # TODO: Decide what to do about p0
    # Find p0 by solving the following:
    # log(p0) = C + log(gammafunc(eta2 + 1) / (-eta1)^(eta2 + 1))
    # p0 = exp(C + log(gamma(eta2 + 1) / (-eta1)^(eta2 + 1)))
    p0 = 0

    params_valid = k > zero(k) && theta > zero(theta)
    if !params_valid
        error_string = "Error: resulting gamma parameters are invalid. Got k = ", k, " and Θ = ", theta, ".
        Input data might not be gamma distributed, or number of bins needs to change.
        Exiting"
        error(error_string)
    end

    return k, theta, p0
end
"""
    get_distr_parameters(beta0, beta1, beta2, distr::Type{Normal})

Gets the parameters which describe a normal distribution based on the
parameters input. Converts the quadratic parameter inputs into a normal
distribution.
"""
function get_distr_parameters(beta0, beta1, beta2, distr::Type{Normal})

    sigma = (-2 * beta2)^-(1/2)

    mu = beta1 * sigma^2

    # TODO: Decide what to do about p0
    p0 = 0

    return mu, sigma, p0
end


"""
    fit_null_distribution(midpoints, counts, num_bins, bin_width, proportion_to_keep, distr, verbose=true)

Fit a distribution to the discretized inputs. Uses mode-matching, since
the input test statistics may have been truncated.
"""
function fit_null_distribution(midpoints, counts, num_bins, bin_width, proportion_to_keep, distr; verbose = true)

    # Remove highest test statistics, that dont fit the zero assumption
    midpoints, counts = keep_bottom_proportion_of_hist(midpoints, counts, proportion_to_keep, verbose)

    # Estimate parameters using regression
    param0, param1, param2 = estimate_fit_parameters(num_bins, bin_width, midpoints, counts, distr)

    # TODO: Decide what to do about p0
    # Calculate distribution parameters
    k, theta, p0 = get_distr_parameters(param0, param1, param2, distr)

    # Create distribution
    f0 = distr(k, theta)

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
    r = linspace(midpoints[1], midpoints[end], length(midpoints))
    scaled_interpolation = Interpolations.scale(interpolation, r)

    fhat(x) = max(scaled_interpolation[x], 0)
    fhat(x::AbstractArray) = [fhat(i) for i in x]

    return fhat
end


"""
    calculate_posterior(test_statistics, priors, null_distr, mixture_pdf, tail, w0=0.0)

Calculate the empirical Bayes posterior using the priors, null distribution and
mixture distribution.

# Arguments
- `test_statistics` : list of test statistics.
- `priors` : one dimensional list of prior values that correspond with the
   test_statistics, such that `priors[i]` is the prior for `test_statistics[i]`.
- `null_distr` : the null distribution of test statistics
- `mixture_pdf` : function representing the mixture distribution of test statistics
- `tail` : Whether the test is two-tailed (:two) or one-tailed (:lower or :upper)
- `w0` : Default constant for the prior calculation
- `null_value = -Inf` : what to set the unwanted tail values to
"""
function calculate_posterior(test_statistics, priors, null_distr, mixture_pdf, tail; w0=0.0, null_value = -Inf)
    @assert length(test_statistics) == size(priors, 1)

    null_pdf(x) = pdf(null_distr, x)

    # Prior function
    prior_fn(x) = exp(w0) / ( exp(w0) + exp(x) )

    num_test_statistics = length(test_statistics)

    # Posterior array
    posterior = Array{Float64}(num_test_statistics)

    for i in 1:num_test_statistics
        ts = test_statistics[i]
        prior_val = prior_fn(priors[i])
        null_val = null_pdf(ts)
        mix_val = mixture_pdf(ts)

        # if mixture distr equals 0, then just return a 0 posterior
        if mix_val == zero(mix_val)
            posterior[i] = 0.0
            continue
        end

        fdr = null_val / mix_val
        p1 = 1 - prior_val * fdr
        posterior[i] = p1
    end

    # Get rid of the lower tail for an upper-tailed test, or vice-versa
    if tail == :upper || tail == :lower
        null_mode = mode(null_distr)
        compare = tail == :upper ? (<=) : (>=)
        zero_indices = find(t -> compare(t, null_mode), test_statistics)
        posterior[zero_indices] = null_value
    end

    return posterior
end


"""
    calculate_posterior(test_statistics, null_distr, mixture_pdf, tail, w0=0.0)

Calculate the empirical Bayes posterior using no priors, null distribution and
mixture distribution.

# Arguments
- `test_statistics` : list of test statistics.
- `null_distr` : the null distribution of test statistics
- `mixture_pdf` : function representing the mixture distribution of test statistics
- `tail` : Whether the test is two-tailed (:two) or one-tailed (:lower or :upper)
- `w0` : Default constant for the prior calculation
- `null_value = -Inf` : what to set the unwanted tail values to
"""
function calculate_posterior(test_statistics, null_distr, mixture_pdf, tail; w0=0.0, null_value = -Inf)
    priors = [0 for _ in test_statistics]
    return calculate_posterior(test_statistics, priors, null_distr, mixture_pdf, tail, w0=w0)
end



"""
    empirical_bayes(test_statistics, priors, num_bins, distr, proportion_to_keep=1.0, tail=:two, w0=0.0)

Calculate the empirical Bayes posteriors of the input statistics using the priors.

# Arguments
- `test_statistics` : list of test statistics.
- `priors` : one dimensional list of prior values that correspond with the
   test_statistics, such that `priors[i]` is the prior for `test_statistics[i]`.
- `num_bins::Integer` : number of uniform width bins to discretize into.
- `distr` : form of the null distribution to be fitted.
- `proportion_to_keep=1.0` : Proportion of lowest test statistics to
   keep when calculating null distribution.
- `tail=:two` : Whether the test is two-tailed (:two) or one-tailed (:lower or :upper)
- `w0` : Default constant for the prior calculation
- `null_value = -Inf` : what to set the unwanted tail values to
"""
function empirical_bayes(test_statistics, priors, num_bins, distr; proportion_to_keep=1.0, tail=:two, w0=0.0, null_value = -Inf)
     midpoints, counts, bin_width = discretize_test_statistics(test_statistics, num_bins)
     null_distr = fit_null_distribution(midpoints, counts, num_bins, bin_width, proportion_to_keep, distr)
     mixture_pdf = fit_mixture_distribution(midpoints, counts, bin_width)
     posteriors = calculate_posterior(test_statistics, priors, null_distr, mixture_pdf, tail, w0=w0, null_value = null_value)
     return posteriors
end
function empirical_bayes(test_statistics, priors, num_bins, distr::Symbol; proportion_to_keep=1.0, tail=:two, w0=0.0, null_value = -Inf)
    return empirical_bayes(test_statistics, priors, num_bins, get_distr(distr); proportion_to_keep=proportion_to_keep, tail=tail, w0=w0, null_value = null_value)
end


"""
    empirical_bayes(test_statistics, num_bins, distr, proportion_to_keep=1.0, tail=:two, w0=0.0)

Calculate the empirical Bayes posteriors of the input statistics with null priors.

# Arguments
- `test_statistics` : list of test statistics.
- `num_bins::Integer` : number of uniform width bins to discretize into.
- `distr` : form of the null distribution to be fitted.
- `proportion_to_keep=1.0` : Proportion of lowest test statistics to
   keep when calculating null distribution.
- `tail=:two` : Whether the test is two-tailed (:two) or one-tailed (:lower or :upper)
- `w0` : Default constant for the prior calculation
- `null_value = -Inf` : what to set the unwanted tail values to
"""
function empirical_bayes(test_statistics, num_bins, distr; proportion_to_keep=1.0, tail=:two, w0=0.0, null_value = -Inf)
    priors = [0 for _ in test_statistics]
    return empirical_bayes(test_statistics, priors, num_bins, distr, proportion_to_keep=proportion_to_keep, tail=tail, w0=w0, null_value = null_value)
end
function empirical_bayes(test_statistics, num_bins, distr::Symbol; proportion_to_keep=1.0, tail=:two, w0=0.0, null_value = -Inf)
    return empirical_bayes(test_statistics, num_bins, get_distr(distr); proportion_to_keep=proportion_to_keep, tail=tail, w0=w0, null_value = null_value)
end

# Utility function for getting a distribution from a symbol, so that the caller needn't
# depend on Distributions.jl
function get_distr(d::Symbol)
    distrs = Dict(
        :Gamma => Gamma,
        :Normal => Normal
    )
    return distrs[d]
end
