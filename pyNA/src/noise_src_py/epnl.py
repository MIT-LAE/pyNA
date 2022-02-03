import pdb
import openmdao
import openmdao.api as om
import numpy as np
from tqdm import tqdm
from pyNA.src.settings import Settings

def epnl(self, t_o: np.ndarray, pnlt: np.ndarray, C: np.ndarray = np.ones(1)) -> np.float64:
    """
    Compute effective perceived noise level.

    :param t_o: Observer time [s]
    :type t_o: np.ndarray [n_t']
    :param pnlt: perceived noise level, tone corrected [PNdB]
    :type pnlt: np.ndarray [n_t']
    :param C: pnlt tone correction [dB]
    :type C: np.ndarray [n_t'] settings.N_f']

    :return: epnl [EPNdB]
    :rtype: Float64
    """
    
    # Load options
    settings = self.options['settings']

    # Interpolate time, pnlt and C
    n_ip = np.int64(np.ceil((t_o[-1]-t_o[0])/0.5))

    t_ip = np.zeros(n_ip)
    for i in np.arange(n_ip):
        t_ip[i] = t_o[0] + i*0.5

    pnlt_ip = np.interp(t_ip, t_o, pnlt)

    # Compute max. PNLT
    pnltm = np.max(pnlt_ip)

    # Check tone band-sharing
    i_max = np.where(pnlt_ip == pnltm)[0][0]
    if settings.bandshare:

        C_ip = np.zeros((n_ip, settings.N_f))
        for j in np.arange(settings.N_f):
            C_ip[:,j] = np.interp(t_ip, t_o, C[:,j])

        if i_max == 0 or i_max == 1:
            i_left_bandshare = 0
        else:
            i_left_bandshare = i_max-2

        if i_max == np.shape(C_ip)[0]-1 or i_max == np.shape(C_ip)[0]-2:
            i_right_bandshare = np.shape(C_ip)[0]
        else:
            i_right_bandshare = i_max + 3

        C_bandshare = np.zeros(i_right_bandshare-i_left_bandshare)
        for i, k in enumerate(np.arange(i_left_bandshare, i_right_bandshare)):
            C_bandshare[i] = np.max(C_ip[k, :])

        if C_bandshare[2] < np.mean(C_bandshare):
            pnltm = pnltm - C_bandshare[2] + np.mean(C_bandshare)

    # Compute max. PNLT point (k_m)
    I = np.where(pnlt_ip > pnltm - 10.)

    # ICAO Annex 16 procedures (p132 - App. 2-18)
    f_int = 10. ** (pnlt_ip / 10.)

    # Compute integration bounds
    if pnltm > 10:
        i_1 = I[0][0]
        if np.abs(pnlt_ip[i_1] - (pnltm - 10)) > np.abs(pnlt_ip[i_1 - 1] - (pnltm - 10)):
            i_1 = i_1 - 1

        i_2 = I[0][-1]
        if i_2 < pnlt_ip.shape[0] - 1:
            if np.abs(pnlt_ip[i_2] - (pnltm - 10)) > np.abs(pnlt_ip[i_2 + 1] - (pnltm - 10)):
                i_2 = i_2 + 1

        D = 10 * np.log10(np.sum(f_int[i_1:i_2])) - pnltm - 10 * np.log10(20)

    else:
        D = 10 * np.log10(np.sum(f_int)) - pnltm - 10 * np.log10(20)

    # Compute EPNL
    epnl = pnltm + D

    return epnl
