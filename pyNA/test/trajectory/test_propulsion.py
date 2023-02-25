import unittest
import openmdao.api as om
from pyNA.src.trajectory_model.propulsion import Propulsion
from openmdao.utils.assert_utils import assert_check_partials
from pyNA.pyna import pyna
import pdb


class TestPropulsion(unittest.TestCase):

	def test_evaluate_stratified(self):

		settings = dict()
		settings['case_name'] = 'stca'
		settings['ac_name'] = 'stca'
		settings['atmosphere_mode'] = 'stratified'
		settings['engine_deck_file_name'] = 'engine_deck_stca.csv'
		settings['fan_inlet_source'] = True
		settings['fan_discharge_source'] = True
		settings['core_source'] = True
		settings['jet_mixing_source'] = True
		settings['jet_shock_source'] = True
		settings['airframe_source'] = True
		settings['core_turbine_attenuation_method'] = 'ge'
		settings['thrust_lapse'] = True

		py = pyna(settings, trajectory_mode='model')
		py.aircraft.engine.get_performance_deck(settings=py.settings)

		nn = 1
		
		prob = om.Problem()
		prob.model.add_subsystem("p", Propulsion(vec_size=nn, extrapolate=True, method='cubic', engine=py.aircraft.engine, atmosphere_mode=py.settings['atmosphere_mode']))
		prob.setup(force_alloc_complex=True)
		
		prob.set_val('p.z', 0.)
		prob.set_val('p.M_0', 0.3)
		prob.set_val('p.tau', 0.8)
		prob.run_model()

		self.assertAlmostEqual(prob.get_val('p.jet_V')[0], 384.05349, 2)
		self.assertAlmostEqual(prob.get_val('p.core_mdot')[0], 29.58988, 2)
		self.assertAlmostEqual(prob.get_val('p.fan_N')[0], 3134.53006, 2)

	def test_evaluate_sealevel(self):
		settings = dict()
		settings['case_name'] = 'stca'
		settings['ac_name'] = 'stca'
		settings['atmosphere_mode'] = 'sealevel'
		settings['engine_deck_file_name'] = 'engine_deck_stca.csv'
		settings['fan_inlet_source'] = True
		settings['fan_discharge_source'] = True
		settings['core_source'] = True
		settings['jet_mixing_source'] = True
		settings['jet_shock_source'] = True
		settings['airframe_source'] = True
		settings['core_turbine_attenuation_method'] = 'ge'
		settings['thrust_lapse'] = False
		settings['F00'] = 80000.

		py = pyna(settings, trajectory_mode='model')
		py.aircraft.engine.get_performance_deck(settings=py.settings)

		nn = 1
		
		prob = om.Problem()
		prob.model.add_subsystem("p", Propulsion(vec_size=nn, extrapolate=True, method='cubic', engine=py.aircraft.engine, atmosphere_mode=py.settings['atmosphere_mode']))
		prob.setup(force_alloc_complex=True)
		
		prob.set_val('p.M_0', 0.3)
		prob.set_val('p.tau', 0.8)
		prob.run_model()

		self.assertAlmostEqual(prob.get_val('p.jet_V')[0], 408.25766, 2)
		self.assertAlmostEqual(prob.get_val('p.core_mdot')[0], 32.00774, 2)
		self.assertAlmostEqual(prob.get_val('p.fan_N')[0], 3419.83905, 2)


	def test_partials(self):

		settings = dict()
		settings['case_name'] = 'stca'
		settings['ac_name'] = 'stca'
		settings['atmosphere_mode'] = 'stratified'
		settings['engine_deck_file_name'] = 'engine_deck_stca.csv'
		settings['fan_inlet_source'] = True
		settings['fan_discharge_source'] = True
		settings['core_source'] = True
		settings['jet_mixing_source'] = True
		settings['jet_shock_source'] = True
		settings['airframe_source'] = True
		settings['core_turbine_attenuation_method'] = 'ge'
		settings['thrust_lapse'] = True

		py = pyna(settings, trajectory_mode='model')
		py.aircraft.engine.get_performance_deck(settings=py.settings)

		nn = 1
		
		prob = om.Problem()
		prob.model.add_subsystem("p", Propulsion(vec_size=nn, extrapolate=True, method='cubic', engine=py.aircraft.engine, atmosphere_mode=py.settings['atmosphere_mode']))
		prob.setup(force_alloc_complex=True)
		
		prob.set_val('p.z', 0.)
		prob.set_val('p.M_0', 0.3)
		prob.set_val('p.tau', 0.8)
		prob.run_model()

		data = prob.check_partials(compact_print=True, method='cs')
		assert_check_partials(data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
	unittest.main()



