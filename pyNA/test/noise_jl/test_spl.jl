using BenchmarkTools
using Test

include("../../src/noise_src_jl/spl.jl")

if true
	@testset "spl module" begin
	    
		## Test 1: single time step
		# Inputs
		spl = [10., 10., 10., 10., 10.]
		rho_0 = 1.
		c_0 = 300.

		f_spl!(spl, rho_0, c_0)

		spl_input = [10., 10., 10., 10., 10.]
		spl_check = spl_input .+ 20*log10(rho_0 * c_0^2)

		@test isapprox(spl, spl_check, rtol=1e-6)
		
	end
end

# Inputs
spl = [10., 10., 10., 10., 10.]
rho_0 = 1.
c_0 = 300.
@btime f_spl!(spl, rho_0, c_0)

