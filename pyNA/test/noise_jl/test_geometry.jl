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
	x = reshape([6030.166565, 6137.090405], (2, 1))
	y = reshape([0.0, 0.0], (2, 1))
	z = reshape([696.938855, 706.6032584], (2, 1))
	alpha = reshape([12.68815999, 12.81889598], (2, 1))
	gamma = reshape([5.272575985, 5.041840023], (2, 1))
	t_s = reshape([72.96, 73.96], (2, 1))
	T_0 = reshape([293.6755732, 293.62345919999996], (2, 1))
	c_0 = reshape([343.5096, 343.47912], (2, 1))
	n_t = 2

	## Test 1: Lateral position
	# Inputs
	if true
		x_obs = [3.75666e+03, 4.50000e+02, 1.21920e+00]

		r = reshape([2419.784688655463, 2523.195510239608], (2,1))
		beta = reshape([16.709123862957814, 16.233934581349075], (2,1))
		theta = reshape([169.24130264292222, 169.63861328834926], (2,1))
		phi = reshape([-94.98540135634528, -97.4320293734841], (2,1))
		c_bar = reshape([349.97216040498137, 349.95858761280715], (2,1))
		t_o = reshape([79.87422050786934, 81.16998312243521], (2,1))

		sol = geometry(settings, x_obs, x, y, z, alpha, gamma, t_s, c_0, T_0)
		@test(isapprox(sol[1], r, rtol=1e-6))
		@test(isapprox(sol[2], beta, rtol=1e-6))
		@test(isapprox(sol[3], theta, rtol=1e-6))
		@test(isapprox(sol[4], phi, rtol=1e-6))
		@test(isapprox(sol[5], c_bar, rtol=1e-6))
		@test(isapprox(sol[6], t_o, rtol=1e-6))
	end

	## Test 2: Flyover position
	# Inputs
	if true
		x_obs = [6.49986e+03, 0.00000e+00, 1.21920e+00]

		r = reshape([839.4270434262999, 793.2013923974913], (2,1))
		beta = reshape([55.975973178949545, 62.78384664532227], (2,1))
		theta = reshape([73.93670915394954, 80.64458264832226], (2,1))
		phi = reshape([-0.0, 0.0], (2,1))
		c_bar = reshape([349.97216040498137, 349.95858761280715], (2,1))
		t_o = reshape([75.35855376626223, 76.22655787419934], (2,1))
		
		sol = geometry(settings, x_obs, x, y, z, alpha, gamma, t_s, c_0, T_0)
		@test(isapprox(sol[1], r, rtol=1e-6))
		@test(isapprox(sol[2], beta, rtol=1e-6))
		@test(isapprox(sol[3], theta, rtol=1e-6))
		@test(isapprox(sol[4], phi, rtol=1e-6))
		@test(isapprox(sol[5], c_bar, rtol=1e-6))
		@test(isapprox(sol[6], t_o, rtol=1e-6))
	end

end

