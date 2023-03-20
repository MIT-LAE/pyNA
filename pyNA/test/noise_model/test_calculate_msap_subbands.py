import unittest
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pyNA.src.noise_model.python.utils.get_msap_subbands import get_msap_subbands
from pyNA.src.noise_model.python.utils.get_frequency_subbands import get_frequency_subbands
from pyNA.src.noise_model.python.utils.get_frequency_bands import get_frequency_bands
import pdb

class TestMSAPSubbands(unittest.TestCase):
    
    def test_evaluate(self):
        
        settings = dict()
        settings['n_frequency_bands'] = 24
        settings['n_frequency_subbands'] = 5

        spl_in = jnp.array([30., 40., 45., 51., 55., 53., 60., 67., 78., 89., 88., 90., 91., 90., 89., 86., 85., 83., 84., 86., 81., 77., 75., 73.])
        msap_in = 10**(spl_in/10.)
        msap_sb = get_msap_subbands(msap_in=msap_in, settings=settings)
        spl_sb = 10*jnp.log10(msap_sb)

        for i in jnp.arange(settings['n_frequency_bands']):
            
            # Compute log-sum of subband spl
            sum_sb = 10*jnp.log10(jnp.sum(10**(spl_sb[i*settings['n_frequency_subbands']:(i+1)*settings['n_frequency_subbands']]/10.)))
            self.assertAlmostEqual(spl_in[i], sum_sb, 3)


    def test_plot_subbands(self):

        if False: 
            settings = dict()
            settings['n_frequency_bands'] = 24
            settings['n_frequency_subbands'] = 5

            spl_in = jnp.array([30., 40., 45., 51., 55., 53., 60., 67., 78., 89., 88., 90., 91., 90., 89., 86., 85., 83., 84., 86., 81., 77., 75., 73.])
            msap_in = 10**(spl_in/10.)
            msap_sb = compute_msap_subbands(msap_in=msap_in, settings=settings)
            spl_sb = 10*jnp.log10(msap_sb)

            f = get_frequency_bands(n_frequency_bands=settings['n_frequency_bands'])
            f_sb = get_frequency_subbands(f=f, n_frequency_subbands=settings['n_frequency_subbands'])
            plt.semilogx(f, spl_in, '.-')
            plt.semilogx(f_sb, spl_sb, '.-')
            plt.show()


    def test_jax(self):
        
        settings = dict()
        settings['n_frequency_bands'] = 1
        settings['n_frequency_subbands'] = 5

        spl_in = jnp.array([80., ])
        msap_in = 10**(spl_in/10.)
        f_prime_0 = jax.jacfwd(get_msap_subbands, argnums=0)
        J = f_prime_0(msap_in, settings)
        self.assertTrue(sum(abs(J-jnp.array([[0.2], [0.2], [0.2], [0.2], [0.2]])))<1e-6)

if __name__ == '__main__':
	unittest.main()
