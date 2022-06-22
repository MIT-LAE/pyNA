using BenchmarkTools
using Test

include("../../src/noise_src_jl/oaspl.jl")


@testset "oaspl module" begin

	## Test 1: logarithmic addition
	# Inputs
	spl = [10., 10.]

	sol = f_oaspl(spl)
	oaspl = 10. + 10*log10(2)
	@test(isapprox(sol, oaspl, rtol=1e-6))

	## Test 2: logarithmic sum of zeros: not equal to zeros!
	# Inputs
	spl = reshape([0., 0.], (1, 2))

	sol = f_oaspl(spl)
	oaspl = 10*log10(2)
	@test(isapprox(sol, oaspl, rtol=1e-6))

	## Test 3: logarithmic sum of nasa stca sound levels 
	spl = [37.1, 44.9, 52.6, 59.3, 65.7, 72.2, 77.7, 82.9, 87.8, 92.5, 96.5,100.3,104.0,107.2,110.2,113.3,115.9,118.3,120.6,122.7,124.3,125.6,126.4,126.7]

	sol = f_oaspl(spl)
	oaspl = 133.0

	@test(isapprox(sol, oaspl, rtol=1e-1))

end

spl = [37.1, 44.9, 52.6, 59.3, 65.7, 72.2, 77.7, 82.9, 87.8, 92.5, 96.5,100.3,104.0,107.2,110.2,113.3,115.9,118.3,120.6,122.7,124.3,125.6,126.4,126.7]
@btime f_oaspl(spl)