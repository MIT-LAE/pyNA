using BenchmarkTools
using Test

include("../../src/noise_src_jl/shielding.jl")

struct Settings_shielding
	case_name
	x_observer_array
	N_f
	shielding
	observer_lst
end

@testset "shielding module" begin

	## Test 1: stca trajectory design case (shielding not implemented yet)
	settings = Settings_shielding("stca", [[6500., 0., 1.2],[3500., 450., 1.2]], 24, true, ("flyover", "lateral"))

	sol = shielding(settings, 10)

	@test(isapprox(sol, zeros(2, 10, settings.N_f)))


end

