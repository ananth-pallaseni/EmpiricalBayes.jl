using EmpiricalBayes
using Base.Test


# write your own tests here
#@test 1 == 2

############################# Discretization ###########################

@testset "Discretization Tests" begin

# Basic functionality
test_test_statistics = [1, 1.5, 1.2, 2, 2.3, 2.7, 3, 3.3, 3.9, 4]
mids, counts, bin_width = discretize_test_statistics(test_test_statistics, 3)
@test mids == [0.5, 1.5, 2.5]
@test counts == [3, 3, 4]
@test bin_width == 1


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
