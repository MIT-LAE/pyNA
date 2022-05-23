using Test
using ReverseDiff
include("../../src/noise_src_jl/smooth_max.jl")


# Inputs
k_smooth = 50
level_int = [1., 2., 3., 4., 5., 4., 3., 2., 1.]

# --- Test 1: evaluate smooth_max
@test smooth_max(k_smooth, level_int) == 5

# --- Test 2: evaluate derivatives of smooth_max
J_sol = [1.3838965267367376e-87, 7.17509597316441e-66, 3.7200759760208366e-44, 1.9287498479639178e-22, 1.0, 1.9287498479639178e-22, 3.7200759760208366e-44, 7.17509597316441e-66, 0.0]

f_smooth_max = (x)->smooth_max(k_smooth, x)
J = zeros(9)

@test ReverseDiff.gradient!(J, f_smooth_max, level_int) == J_sol