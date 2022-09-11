import pdb
import openmdao
import openmdao.api as om
import numpy as np
from tqdm import tqdm


def ioaspl(self, t_o: np.ndarray, oaspl: np.ndarray) -> np.float64:
    """
    Compute time-integrated oaspl.

    :param t_o: observer time [s]
    :type t_o: np.ndarray [n_t']]
    :param oaspl: overall sound pressure level [dB]
    :type oaspl: np.ndarray [n_t']]

    :return: ioaspl
    :rtype: Float64
    """
    # Load settings
    n_t = self.options['n_t']

    # Trapezoidal integration
    ioaspl = 0
    for i in tqdm(np.arange(n_t-1), desc='Integrated oaspl'):
        ioaspl = ioaspl + (t_o[i+1]-t_o[i])*0.5*(oaspl[i+1]+oaspl[i])

    return ioaspl
