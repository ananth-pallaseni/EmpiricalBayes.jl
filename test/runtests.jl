using EmpiricalBayes
using Base.Test


# write your own tests here
#@test 1 == 2

############################# Discretization ###########################

@testset "Discretization Tests" begin

# Basic functionality
test_test_statistics = [1, 1.5, 1.2, 2, 2.3, 2.7, 3, 3.3, 3.9, 4]
mids, counts, bin_width = discretize_test_statistics(test_test_statistics, 3)
@test mids == [1.5, 2.5, 3.5]
@test counts == [3, 3, 4]
@test bin_width == 1

@testset "Random tests" begin
# Random tests
num_random_tests = 100
for _ in 1:num_random_tests
    bin_width = rand() * 100
    num_bins = Int(round(rand()*100))
    min_ts = rand() * 10
    max_ts = min_ts + bin_width * num_bins

    target_counts = [Int(round(rand()*100)) for i in 1:num_bins]
    target_mids = [min_ts + bin_width/2 + (i-1)*bin_width for i in 1:num_bins]

    rand_test_stats = [[bin_width * (j-1) + min_ts + rand() * bin_width for i in 1:target_counts[j]] for j in 1:length(target_counts)]
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

end





############################# Fit Null ###############################
@testset "Null Fit Tests" begin

end



############################# Fit Mixture ############################
@testset "Mixture Fit Tests" begin

end


############################# Calculate Posterior ####################
@testset "Posterior Tests" begin

end
