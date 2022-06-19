using BenchmarkTools
using Test
using Random

include("../../src/noise_src_jl/split_subbands.jl")

struct Settings_split
	N_b
	N_f
end
settings = Settings_split(5, 24)

if true
	@testset "split subbands module" begin
	    
		## Test 1: constant spl
		if true
			spl = ones(settings.N_f, )

			sol = split_subbands(settings, spl)
		    spl_sb = 0.2*ones(settings.N_f*settings.N_b, )

		    @test(isapprox(sol, spl_sb, rtol=1e-6))

		end

		## Test 2: random spl
		if true
			spl = 10*rand(24)
			sol = split_subbands(settings, spl)

			# Sum of the solution must match input
			check = zeros(settings.N_f, )
			for i in 1:1:settings.N_f
				check[i] = sum(sol[(i-1)*settings.N_b+1:i*settings.N_b])
			end

			@test(isapprox(check, spl, rtol=1e-6))

		end
	end
end

spl = 10*rand(24)
@btime split_subbands(settings, spl)
@btime split_subbands2(settings, spl)