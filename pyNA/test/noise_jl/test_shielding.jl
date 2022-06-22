using BenchmarkTools
using Test
using CSV
using DataFrames

include("../../src/noise_src_jl/shielding.jl")


struct Settings_shielding
	case_name
	N_f
	shielding
end
settings = Settings_shielding("stca", 24, true)

struct Data_shield
	shield_l
	shield_f
	shield_a
end

data_shield_l = DataFrame(CSV.File("../../cases/nasa_stca_standard/shielding/shielding_l.csv"))
data_shield_f = DataFrame(CSV.File("../../cases/nasa_stca_standard/shielding/shielding_f.csv"))
data_shield_a = DataFrame(CSV.File("../../cases/nasa_stca_standard/shielding/shielding_a.csv"))
shield_l = Array(data_shield_l)[:, 2:end]
shield_f = Array(data_shield_f)[:, 2:end]
shield_a = Array(data_shield_a)[:, 2:end]
data = Data_shield(shield_l, shield_f, shield_a)

@testset "shielding module" begin

	## Test 1: stca trajectory design case (shielding not implemented yet)
	@test(isapprox(shielding(settings, data, 10, "lateral"), zeros(settings.N_f,)))

end

## Test 2: 
@btime shielding(settings, data, 10, "lateral")


