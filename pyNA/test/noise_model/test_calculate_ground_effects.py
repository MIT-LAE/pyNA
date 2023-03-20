import jax.numpy as jnp
import unittest
import jax
import jax.numpy as jnp
from pyNA.src.noise_model.python.utils.get_frequency_bands import get_frequency_bands
from pyNA.src.noise_model.python.utils.get_frequency_subbands import get_frequency_subbands
from pyNA.src.noise_model.python.propagation.calculate_ground_effects import calculate_ground_effects
import pdb


class TestGroundEffects(unittest.TestCase):

	def test_evaluate(self):
	
		settings = dict()
		settings['n_frequency_bands'] = 24
		settings['n_frequency_subbands'] = 5
		settings['ground_resistance'] = 291.0 * 515.379
		settings['incoherence_constant'] = 0.01
		
		msap_sb = 1
		rho_0 = 1.225
		c_bar = 340.
		r = 3000.
		beta = 30.
		
		x_obs = jnp.array([3000., 450., 1.2])

		f = get_frequency_bands(settings['n_frequency_bands'])
		f_sb = get_frequency_subbands(f, settings['n_frequency_subbands'])
		
		msap_ge = calculate_ground_effects(msap_sb, rho_0, c_bar, r, beta, f_sb, x_obs, settings)
		
		pdb.set_trace()


	def test_jax(self):
		
		pass


if __name__ == '__main__':
	unittest.main()
