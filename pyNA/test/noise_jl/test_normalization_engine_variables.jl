using BenchmarkTools
using Test

include("../../src/noise_src_jl/normalization_engine_variables.jl")

@testset "normalization engine variables module" begin
    
	# Inputs
	struct Settings_norm
		A_e
	end
	settings = Settings_norm(1.)

	V_j = reshape([600., 600.], (2, 1))
	rho_j = reshape([5., 5.], (2, 1))
	A_j = reshape([2., 2.], (2, 1))
	Tt_j = reshape([300., 300.], (2, 1))

	mdoti_c = reshape([10., 10.], (2, 1))
	Tti_c = reshape([600., 600.], (2, 1))
	Ttj_c = reshape([600., 600.], (2, 1))
	Pti_c = reshape([5e5, 5e5], (2, 1))
	DTt_des_c = reshape([1000., 1000.], (2, 1))
	rho_te_c = reshape([2., 2.], (2, 1))
	c_te_c = reshape([300., 300.], (2, 1))
	rho_ti_c = reshape([2., 2.], (2, 1))
	c_ti_c = reshape([300., 300.], (2, 1))

	DTt_f = reshape([1000., 1000.], (2, 1))
	mdot_f = reshape([10., 10.], (2, 1))
	N_f = reshape([1000., 1000.], (2, 1))
	A_f = reshape([1., 1.], (2, 1))
	d_f = reshape([1., 1.], (2, 1))

	c_0 = reshape([300., 300.], (2, 1))
	rho_0 = reshape([1., 1.], (2, 1))
	T_0 = reshape([300., 300.], (2, 1))
	p_0 = reshape([1e5, 1e5], (2, 1)) 

	## Test 1: Jet mixing
	sol = normalization_jet_mixing(settings, V_j, rho_j, A_j, Tt_j, c_0, rho_0, T_0)
	@test(isapprox(sol[1], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[2], reshape([5., 5.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[3], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[4], reshape([1., 1.], (2, 1)), rtol=1e-6))

	### Test 2: Jet shock
	sol = normalization_jet_shock(settings, V_j, A_j, Tt_j, c_0, T_0)
	@test isapprox(sol[1], reshape([2., 2.], (2, 1)), rtol=1e-6)
	@test isapprox(sol[2], reshape([2., 2.], (2, 1)), rtol=1e-6)
	@test isapprox(sol[3], reshape([1., 1.], (2, 1)), rtol=1e-6)

	### Test 3: Jet
	sol = normalization_jet(settings, V_j, rho_j, A_j, Tt_j, c_0, rho_0, T_0)
	@test(isapprox(sol[1], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[2], reshape([5., 5.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[3], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[4], reshape([1., 1.], (2, 1)), rtol=1e-6))

	### Test 4: Core GE
	sol = normalization_core_ge(settings, mdoti_c, Tti_c, Ttj_c, Pti_c, DTt_des_c, c_0, rho_0, T_0, p_0)
	@test(isapprox(sol[1], reshape([10/300., 10/300], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[2], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[3], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[4], reshape([5., 5.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[5], reshape([1000/300., 1000/300.], (2, 1)), rtol=1e-6))

	### Test 5: Core PW
	sol = normalization_core_pw(settings, mdoti_c, Tti_c, Ttj_c, Pti_c, rho_te_c, c_te_c, rho_ti_c, c_ti_c, c_0, rho_0, T_0, p_0)
	@test(isapprox(sol[1], reshape([10/300., 10/300], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[2], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[3], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[4], reshape([5., 5.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[5], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[6], reshape([1., 1.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[7], reshape([2., 2.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[8], reshape([1., 1.], (2, 1)), rtol=1e-6))

	### Test 6: Fan
	sol = normalization_fan(settings, DTt_f ,mdot_f, N_f, A_f, d_f, c_0, rho_0, T_0)
	@test(isapprox(sol[1], reshape([1000/300., 1000/300], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[2], reshape([10/300, 10/300], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[3], reshape([1000/300/60, 1000/300/60], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[4], reshape([1., 1.], (2, 1)), rtol=1e-6))
	@test(isapprox(sol[5], reshape([1., 1.], (2, 1)), rtol=1e-6))

end





















