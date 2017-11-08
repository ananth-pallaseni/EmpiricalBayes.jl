using EmpiricalBayes
using Base.Test

using Distributions
using Distances

data_folder_path = joinpath(dirname(@__FILE__), "data")

struct TestDist <: ContinuousUnivariateDistribution
    x::Int
end

Distributions.pdf(d::TestDist, n) = d.x
Distributions.mode(d::TestDist) = d.x

############################# Discretization ###########################

@testset "Discretization Tests" begin

# Basic functionality
test_test_statistics = [1, 1.5, 1.2, 2, 2.3, 2.7, 3, 3.3, 3.9, 4]
mids, counts, bin_width = discretize_test_statistics(test_test_statistics, 3)
@test mids == [1.5, 2.5, 3.5]
@test counts == [3, 3, 4]
@test bin_width == 1

@testset "Discretization Random Tests" begin
# Random tests
num_random_tests = 100
for _ in 1:num_random_tests
    bin_width = rand() * 100 + eps()
    num_bins = Int(round(rand()*100)) + 1
    min_ts = rand() * 10
    max_ts = min_ts + bin_width * num_bins

    target_counts = [Int(round(rand()*100)) for i in 1:num_bins]
    target_mids = [min_ts + bin_width/2 + (i-1)*bin_width for i in 1:num_bins]

    rand_test_stats = [[bin_width * (j-1) + min_ts + eps() + rand() * bin_width * 0.9 for i in 1:target_counts[j]] for j in 1:length(target_counts)]
    rand_test_stats = reduce(vcat, rand_test_stats)
    rand_test_stats = [float(x) for x in rand_test_stats]
    push!(rand_test_stats, min_ts)
    push!(rand_test_stats, max_ts)
    target_counts[1] += 1
    target_counts[end] += 1

    rmids, rcounts, rwidths = discretize_test_statistics(rand_test_stats, num_bins)

    @test isapprox(rmids, target_mids, atol=0.00001)
    @test rcounts == target_counts
    @test isapprox(rwidths, bin_width, atol=0.00001)
end
end

# Test silent type conversion
mids, counts, bin_width = discretize_test_statistics(test_test_statistics, 3)
fmids, fcounts, fwidth = discretize_test_statistics(test_test_statistics, 3.0)
@test fmids == mids
@test fcounts == counts
@test fwidth == bin_width

# Test invalid arguments
@test_throws ErrorException discretize_test_statistics(test_test_statistics, 3.3)
@test_throws MethodError discretize_test_statistics("1, 1.5, 1.2, 2, 2.3, 2.7, 3, 3.3, 3.9, 4", 3)

end


############################# Truncating ###########################

@testset "Truncate histogram tests" begin

######## Basic Test
mids, counts, bin_width = discretize_test_statistics(collect(Float64, 1:100), 10)

proportions = [0.89, 0.9, 0.91, 1.0]
targets = [8, 9, 9, 10]

for i in 1:length(proportions)
    @test EmpiricalBayes.keep_bottom_proportion_of_hist(mids, counts, proportions[i], true) == (mids[1:targets[i]], counts[1:targets[i]])
end

end


############################# Fit Null ###############################

@testset "Null Fit Tests" begin

######## Basic test

# Gamma distribution
mids, counts, width = discretize_test_statistics([1, 1.5, 1.2, 2, 2,2,2,2,2, 2.3, 2.7, 3, 3.3, 3.9, 4], 3)

dist = fit_null_distribution(mids, counts, 3, width, 1.0, Gamma, verbose = false)
f(x) = pdf(dist, x)

basic_test_result = f.(0:0.1:10)

basic_test_target = [0.0       ,2.32616e-10,1.22024e-7 ,4.043e-6   ,4.32361e-5 ,0.000248819,0.000967622,0.0028712  ,0.00698951 ,0.0146273  ,0.0271695  ,0.0458241  ,0.0713679  ,0.103957   ,0.143041   ,0.187386   ,0.235202   ,0.284332   ,0.332473   ,0.377397   ,0.417133   ,0.450104   ,0.47521    ,0.491851   ,0.499911   ,0.499696   ,0.491862   ,0.477317   ,0.457138   ,0.43248    ,0.404504   ,0.374322   ,0.342946   ,0.311267   ,0.280033   ,0.249848   ,0.221176   ,0.194349   ,0.169582   ,0.14699,0.126606   ,0.108396   ,0.0922763  ,0.0781275  ,0.0658054  ,0.0551525  ,0.0460054  ,0.0382016  ,0.0315839  ,0.026004   ,0.0213245  ,0.01742    ,0.014178   ,0.0114984  ,0.00929344 ,0.00748661 ,0.00601195 ,0.004813   ,0.00384178 ,0.0030578  ,0.00242711 ,0.00192136 ,0.00151708 ,0.00119488 ,0.000938834,0.000735927,0.000575565,0.000449155,0.000349759,0.000271794,0.000210783,0.000163147,0.000126036,9.71871e-5 ,7.48064e-5 ,5.74787e-5 ,4.40894e-5 ,3.37629e-5 ,2.58132e-5 ,1.97041e-5 ,1.50178e-5,1.14288e-5,8.68487e-6,6.59029e-6,4.9939e-6 ,3.77907e-6,2.85596e-6,2.15554e-6,1.62483e-6,1.22327e-6,9.19837e-7,6.9085e-7 ,5.18267e-7,3.88356e-7,2.90686e-7,2.17344e-7,1.62333e-7,1.2112e-7 ,9.02779e-8,6.72221e-8,5.00055e-8 ]

@test isapprox(basic_test_result, basic_test_target, atol=0.00001)

# Normal distribution

# Test statistics generated from Normal(0, 1) distribution
ts = randn(1000)
mids, counts, width = discretize_test_statistics(ts, 10)
dist = fit_null_distribution(mids, counts, 10, width, 1.0, Normal, verbose = false)
@test isapprox(mean(dist), 0, atol = 0.5)
@test isapprox(std(dist), 1, atol = 0.5)


######## Random Tests
@testset "Null Random Tests" begin

num_random_tests = 300
hellinger_sum = 0
for _ in 1:num_random_tests
    α = Int(round(rand()*10)) + 3
    β = Int(round(rand()*10)) + 0.01
    reference_distr = Gamma(α, β)
    rand_test_stats = rand(reference_distr, 1000)

    num_bins = 3 + Int(round(rand()*100))
    rmids, rcounts, rwidths = discretize_test_statistics(rand_test_stats, num_bins)

    test_distr = fit_null_distribution(rmids, rcounts, num_bins, rwidths, 1.0, Gamma, verbose = false)
    test_pdf(x) = pdf(test_distr, x)

    ref_values = [pdf(reference_distr, x) for x in 0:0.01:20]
    test_values = [test_pdf(x) for x in 0:0.01:20]
    hellinger_dist = hellinger(ref_values, test_values)

    @test hellinger_dist < 0.5

    hellinger_sum += hellinger_dist
end

hellinger_sum = hellinger_sum / num_random_tests
@test hellinger_sum <= 0.3

end

######## Test invalid input error
# Test error out on Gamma inputs where α is small
gamma_distr = Gamma(0.001, 5.0)
ref_test_stats = rand(gamma_distr, 100)
tmids, tcounts, twidths = discretize_test_statistics(ref_test_stats, 10)
@test_throws ErrorException fit_null_distribution(tmids, tcounts, 10, twidths, 1.0, Gamma, verbose = false)

end

############################# Fit Mixture ############################
@testset "Mixture Fit Tests" begin

######## Basic Test - fit spline to standard normal
ref_distr = Normal(0,1);
normal_samples_filepath = joinpath(data_folder_path, "normal_01_1000_samples.txt")
test_stats = readdlm(normal_samples_filepath)
test_mids, test_counts, test_width = discretize_test_statistics(test_stats, 10);
fh = fit_mixture_distribution(test_mids, test_counts, test_width);
test_vals = [fh(x) for x in -5:0.01:5];
ref_vals = [pdf(ref_distr, x) for x in -5:0.01:5];
hellinger_dist = hellinger(test_vals, ref_vals)
@test hellinger_dist < 0.2


end


############################# Calculate Posterior ####################
@testset "Posterior Tests" begin

######## Basic Test
test_post = calculate_posterior([0, 0], [0, 1], TestDist(1), y->1, :two)
@test test_post ≈ [0.5, 0.7310585786300049] atol=0.0001

test_post = calculate_posterior([0, 0], [0, 1], TestDist(2), y->1, :two)
@test test_post ≈ [0.0, 0.4621171572600098] atol=0.0001

test_post = calculate_posterior([1, 2], [0, 1], TestDist(1), y->1, :two)
@test test_post ≈ [0.5, 0.7310585786300049] atol=0.0001

test_post = calculate_posterior([1, 2], [0, 1], TestDist(1), y->0, :two)
@test test_post ≈ [0.0, 0.0] atol=0.0001

test_priors = [0, 1, 1, 0, 1]
test_statistics = [1, 2, 3, 4, 5]
test_null_dist = Normal(2, 5)
test_mix_pdf(x) = pdf(Normal(3, 5), x)
reference_post = [0.469082, 0.725626, 0.736384, 0.529118, 0.756652]
test_post = calculate_posterior(test_statistics, test_priors, test_null_dist, test_mix_pdf, :two)
@test test_post ≈ reference_post atol=0.000001

# No prior
test_post = calculate_posterior([0, 0], TestDist(1), y->1, :two)
@test test_post ≈ [0.5, 0.5] atol=0.0001

# One-tailed test, upper tail
reference_post = [-Inf, -Inf, 0.736384, 0.529118, 0.756652]
test_post = calculate_posterior(test_statistics, test_priors, test_null_dist, test_mix_pdf, :upper)
@test test_post ≈ reference_post atol=0.000001

# One-tailed test, lower tail
reference_post = [0.469082, -Inf, -Inf, -Inf, -Inf]
test_post = calculate_posterior(test_statistics, test_priors, test_null_dist, test_mix_pdf, :lower)
@test test_post ≈ reference_post atol=0.000001

end



############################# empirical_bayes ####################
@testset "Pipeline Tests" begin
# Test that the function at least runs
gamma_stats_filepath = joinpath(data_folder_path, "gamma_51_1000_samples.txt")
gamma_stats = readdlm(gamma_stats_filepath)

priors_filepath = joinpath(data_folder_path, "priors_random_1000_samples.txt")
priors = readdlm(priors_filepath)

eb_at_least_runs = true
try
eb = empirical_bayes(gamma_stats, priors, 10.0, Gamma)
catch
eb_at_least_runs = false
end
@test eb_at_least_runs
# With symbol for distribution
try
eb = empirical_bayes(gamma_stats, priors, 10.0, :Gamma)
catch
eb_at_least_runs = false
end
@test eb_at_least_runs


# Test that it runs without priors
no_priors_eb_at_least_runs = true
try
eb = empirical_bayes(gamma_stats, 10.0, Gamma)
catch
no_priors_eb_at_least_runs = false
end
@test no_priors_eb_at_least_runs
# With symbol for distribution
try
eb = empirical_bayes(gamma_stats, 10.0, :Gamma)
catch
no_priors_eb_at_least_runs = false
end
@test no_priors_eb_at_least_runs

end
