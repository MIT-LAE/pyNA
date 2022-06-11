using BenchmarkTools
using Test

include("../../src/noise_src_jl/spl.jl")

@testset "spl module" begin
    
	## Test 1: single time step
	if true
		# Inputs
		msap_prop = reshape([10., 10., 10., 10., 10.], (1, 5)) 
		rho_0 = [1.]
		c_0 = [300.]

		sol = f_spl(msap_prop, rho_0, c_0)
		spl = msap_prop .+ 20*log10.(rho_0 .* c_0.^2)

		@test isapprox(sol, spl, rtol=1e-6)
	end

	## Test 2: two time steps
	if true
		# Inputs
		msap_prop = reshape([10., 10., 10., 10., 10., 10., 10., 10., 10., 10.], (2, 5)) 
		rho_0 = reshape([1., 1.], (2, 1))
		c_0 = reshape([300., 300.], (2, 1))

		sol = f_spl(msap_prop, rho_0, c_0)
		spl = msap_prop .+ 20*log10.(rho_0 .* c_0.^2)

		@test isapprox(sol, spl, rtol=1e-6)
	end

end

