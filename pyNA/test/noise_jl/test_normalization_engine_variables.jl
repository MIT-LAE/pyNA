using BenchmarkTools
using Test

include("../../src/noise_src_jl/normalization_engine_variables.jl")

# Inputs
struct Settings_norm
	A_e
end
settings = Settings_norm(1.)

c_0 = 300.
rho_0 = 1.
T_0 = 300.
p_0 = 1e5

V_j = 600.
rho_j = 5.
A_j = 2.
Tt_j = 300.

mdoti_c = 10.
Tti_c = 600.
Ttj_c = 600.
Pti_c = 5e5
DTt_des_c = 1000.

mdoti_c = 10.
Tti_c = 600.
Ttj_c = 600.
Pti_c = 5e5
rho_te_c = 2.
c_te_c = 300.
rho_ti_c = 2.
c_ti_c = 300.

DTt_f = 1000.
mdot_f = 10.
N_f = 1000.
A_f = 1.
d_f = 1.

@testset "normalization engine variables module" begin
    
	## Test 1: Jet mixing
	sol = normalization_jet_mixing(settings, V_j, rho_j, A_j, Tt_j, c_0, rho_0, T_0)
	@test(isapprox(sol[1], 2., rtol=1e-6))
	@test(isapprox(sol[2], 5., rtol=1e-6))
	@test(isapprox(sol[3], 2., rtol=1e-6))
	@test(isapprox(sol[4], 1., rtol=1e-6))

	### Test 2: Jet shock
	sol = normalization_jet_shock(settings, V_j, A_j, Tt_j, c_0, T_0)
	@test isapprox(sol[1], 2., rtol=1e-6)
	@test isapprox(sol[2], 2., rtol=1e-6)
	@test isapprox(sol[3], 1., rtol=1e-6)

	### Test 3: Jet
	sol = normalization_jet(settings, V_j, rho_j, A_j, Tt_j, c_0, rho_0, T_0)
	@test(isapprox(sol[1], 2., rtol=1e-6))
	@test(isapprox(sol[2], 5., rtol=1e-6))
	@test(isapprox(sol[3], 2., rtol=1e-6))
	@test(isapprox(sol[4], 1., rtol=1e-6))

	### Test 4: Core GE
	sol = normalization_core_ge(settings, mdoti_c, Tti_c, Ttj_c, Pti_c, DTt_des_c, c_0, rho_0, T_0, p_0)
	@test(isapprox(sol[1], 10/300., rtol=1e-6))
	@test(isapprox(sol[2], 2., rtol=1e-6))
	@test(isapprox(sol[3], 2., rtol=1e-6))
	@test(isapprox(sol[4], 5., rtol=1e-6))
	@test(isapprox(sol[5], 1000/300., rtol=1e-6))

	### Test 5: Core PW
	sol = normalization_core_pw(settings, mdoti_c, Tti_c, Ttj_c, Pti_c, rho_te_c, c_te_c, rho_ti_c, c_ti_c, c_0, rho_0, T_0, p_0)
	@test(isapprox(sol[1], 10/300., rtol=1e-6))
	@test(isapprox(sol[2], 2., rtol=1e-6))
	@test(isapprox(sol[3], 2., rtol=1e-6))
	@test(isapprox(sol[4], 5., rtol=1e-6))
	@test(isapprox(sol[5], 2., rtol=1e-6))
	@test(isapprox(sol[6], 1., rtol=1e-6))
	@test(isapprox(sol[7], 2., rtol=1e-6))
	@test(isapprox(sol[8], 1., rtol=1e-6))

	### Test 6: Fan
	sol = normalization_fan(settings, DTt_f ,mdot_f, N_f, A_f, d_f, c_0, rho_0, T_0)
	@test(isapprox(sol[1], 1000/300., rtol=1e-6))
	@test(isapprox(sol[2], 10/300, rtol=1e-6))
	@test(isapprox(sol[3], 1000/300/60, rtol=1e-6))
	@test(isapprox(sol[4], 1., rtol=1e-6))
	@test(isapprox(sol[5], 1., rtol=1e-6))
end


@btime normalization_jet_mixing(settings, V_j, rho_j, A_j, Tt_j, c_0, rho_0, T_0)
@btime sol = normalization_jet_shock(settings, V_j, A_j, Tt_j, c_0, T_0)
@btime sol = normalization_jet(settings, V_j, rho_j, A_j, Tt_j, c_0, rho_0, T_0)
@btime sol = normalization_core_ge(settings, mdoti_c, Tti_c, Ttj_c, Pti_c, DTt_des_c, c_0, rho_0, T_0, p_0)
@btime sol = normalization_core_pw(settings, mdoti_c, Tti_c, Ttj_c, Pti_c, rho_te_c, c_te_c, rho_ti_c, c_ti_c, c_0, rho_0, T_0, p_0)
@btime sol = normalization_fan(settings, DTt_f ,mdot_f, N_f, A_f, d_f, c_0, rho_0, T_0)
