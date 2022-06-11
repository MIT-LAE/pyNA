using BenchmarkTools
using Test
using Random

include("../../src/noise_src_jl/split_subbands.jl")

struct Settings_split
	N_b
	N_f
end
settings = Settings_split(5, 24)

@testset "split subbands module" begin
    
	## Test 1: constant msap
	if true
		msap = ones(2, settings.N_f)

		sol = split_subbands(settings, msap)
	    msap_sb = 0.2*ones(2, settings.N_f*settings.N_b)

	    @test(isapprox(sol, msap_sb, rtol=1e-6))

	end

	## Test 2: ramp msap
	if true
		msap = reshape(range(1, 24, length=settings.N_f), (1, settings.N_f))
		sol = split_subbands(settings, msap)

		# Sum of the solution must match input
		check = zeros(1, settings.N_f)
		for i in 1:1:settings.N_f
			check[i] = sum(sol[(i-1)*settings.N_b+1:i*settings.N_b])
		end

		@test(isapprox(check, msap, rtol=1e-6))

	end

	## Test 3: random msap
	if true
		msap = reshape(10*rand(24), (1, settings.N_f))
		sol = split_subbands(settings, msap)

		# Sum of the solution must match input
		check = zeros(1, settings.N_f)
		for i in 1:1:settings.N_f
			check[i] = sum(sol[(i-1)*settings.N_b+1:i*settings.N_b])
		end

		@test(isapprox(check, msap, rtol=1e-6))

	end

end
