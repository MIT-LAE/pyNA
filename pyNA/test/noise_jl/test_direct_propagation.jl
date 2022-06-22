using BenchmarkTools
using Test

include("../../src/noise_src_jl/direct_propagation.jl")

struct Settings_dp
	r_0
end
settings = Settings_dp(0.3048)


@testset "direct propagation module" begin
	
	## Test 1: A-weighting with delta-dB weights (example NASA STCA Standard)	
	r = 100.
	I_0 = 410.
	spl = ones(24,)
	direct_propagation!(spl, settings, r, I_0)
	sol = [9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6, 9.284412587707317e-6]
	@test(isapprox(spl, sol, rtol=1e-6))
end

## Test 2: timing
r = 100.
I_0 = 410.
spl = ones(24,)
@btime direct_propagation!(spl, settings, r, I_0)