import pdb
import openmdao
import openmdao.api as om
import numpy as np
from tqdm import tqdm
from pyNA.src.settings import Settings


def ipnlt(self, t_o: np.ndarray, pnlt: np.ndarray) -> np.float64:
    """
    Compute time-integrated pnlt.

    :param t_o: observer time [s]
    :type t_o: np.ndarray [subset]
    :param pnlt: perceived noise leve, tone-corrected [PNdB]
    :type pnlt: np.ndarray [subset]

    :return: ipnlt
    :rtype: Float64
    """

    # Interpolate time, pnlt and C
    n_ip = np.int64(np.ceil((t_o[-1] - t_o[0]) / 0.5))

    t_ip = np.zeros(n_ip)
    for i in np.arange(n_ip):
        t_ip[i] = t_o[0] + i * 0.5

    pnlt_ip = np.interp(t_ip, t_o, pnlt)

    # Compute max. PNLT
    pnltm = np.max(pnlt_ip)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10. ** (pnlt_ip / 10.)

    D = 10 * np.log10(np.sum(f_int)) - pnltm - 10 * np.log10(20)

    # Compute EPNL
    ipnlt = pnltm + D

    return ipnlt
