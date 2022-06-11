using BenchmarkTools
using Test

include("../../src/noise_src_jl/ilevel.jl")

@testset "ilevel module" begin
    
    ## Test 1: constant level
	t_o = range(0, 10.0, length=21)
	level = ones(21)
	
	sol = f_ilevel(t_o, level)
	ilevel = 1.
	@test isapprox(sol, ilevel, rtol=1e-6)


	## Test 2: ramp
	t_o = range(0, 10, length=21)
	level = t_o

	sol = f_ilevel(t_o, level)
	ilevel = 10 * log10(sum(10 .^(level[1:end-1] / 10.))) - 10 * log10(10. / 0.5)

	@test isapprox(sol, ilevel, rtol=1e-6)

end

