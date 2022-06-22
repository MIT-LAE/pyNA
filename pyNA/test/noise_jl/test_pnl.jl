using BenchmarkTools
using Random
using Test

include("../../src/noise_src_jl/pnl.jl")


@testset "pnl module" begin
    
	## Test 1
	N = 10*rand(24)
	sol = f_pnl(N)
	@test(isapprox(f_pnl(N), sol, rtol=1e-6))

end

## Test 2: timing
N = 10*rand(24)
@btime f_pnl(N)
