using BenchmarkTools
using Test

include("../../src/noise_src_jl/geometry.jl")

@testset "Geometry module" begin
    
	# Settings struct
	struct Settings_geom
	    dT::Float64
	end
	settings = Settings_geom(10.0169)

	# Inputs
	x = 6030.166565
	y = 0.0
	z = 696.938855
	alpha = 12.68815999
	gamma = 5.272575985
	t_s = 72.96
	T_0 = 293.6755732
	c_0 = 343.5096

	## Test 1: Lateral position
	x_obs = [3.75666e+03, 4.50000e+02, 1.21920e+00]

	r = 2419.784688655463
	beta = 16.709123862957814
	theta = 169.24130264292222
	phi = -94.98540135634528
	c_bar = 343.5096
	t_o = 80.00430003893767

	sol = geometry(x_obs, x, y, z, alpha, gamma, t_s, c_0, T_0)
	@test(isapprox(sol[1], r, rtol=1e-6))
	@test(isapprox(sol[2], beta, rtol=1e-6))
	@test(isapprox(sol[3], theta, rtol=1e-6))
	@test(isapprox(sol[4], phi, rtol=1e-6))
	@test(isapprox(sol[5], c_bar, rtol=1e-6))
	@test(isapprox(sol[6], t_o, rtol=1e-6))

	## Test 2: Flyover position
	x_obs = [6.49986e+03, 0.00000e+00, 1.21920e+00]

	r = 839.4270434262999
	beta = 55.975973178949545
	theta = 73.93670915394954
	phi = 0.0
	c_bar = 343.5096
	t_o = 75.40367855636727
	
	sol = geometry(x_obs, x, y, z, alpha, gamma, t_s, c_0, T_0)
	@test(isapprox(sol[1], r, rtol=1e-6))
	@test(isapprox(sol[2], beta, rtol=1e-6))
	@test(isapprox(sol[3], theta, rtol=1e-6))
	@test(isapprox(sol[4], phi, rtol=1e-6))
	@test(isapprox(sol[5], c_bar, rtol=1e-6))
	@test(isapprox(sol[6], t_o, rtol=1e-6))

end

## Test 3: timing
x_obs = [3.75666e+03, 4.50000e+02, 1.21920e+00]
x = 6030.166565
y = 0.0
z = 696.938855
alpha = 12.68815999
gamma = 5.272575985
t_s = 72.96
T_0 = 293.6755732
c_0 = 343.5096
@btime geometry(x_obs, x, y, z, alpha, gamma, t_s, c_0, T_0)

