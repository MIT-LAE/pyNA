using BenchmarkTools
using Test

include("../../src/noise_src_jl/spl.jl")

@testset "spl module" begin
    
	## Test 1
	spl = 10*ones(24,)
	rho_0 = 1.
	c_0 = 300.

	f_spl!(spl, rho_0, c_0)

	spl_input = 10*ones(24,)
	spl_check = spl_input .+ 20*log10(rho_0 * c_0^2)

	@test isapprox(spl, spl_check, rtol=1e-6)
		
end

## Test 2: timing
spl = 10*ones(24,)
rho_0 = 1.
c_0 = 300.
@btime f_spl!(spl, rho_0, c_0)

