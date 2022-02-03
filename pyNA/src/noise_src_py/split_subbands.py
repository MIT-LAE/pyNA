import numpy as np
from typing import Dict, Any


def split_subbands(settings: Dict[str, Any], msap_in: np.ndarray) -> np.ndarray:
    """
    Compute subfrequency bands for a given 1/3rd octave frequency spectrum.

    :param settings: pyna settings
    :type settings: Dict[str, Any]
    :param msap_in: mean-square acoustic pressure of the source (re. rho_0,^2c_0^2) [-]
    :type msap_in: np.ndarray [settings.N_f']]

    :return: mean-square acoustic pressure of the source, split into sub-frequency bands (re. rho_0,^2c_0^2) [-]
    :rtype: msap_sb [settings.N_f']*settings.N_b']]

    """

    # Integer [-]
    m = (settings.N_b - 1) / 2

    # Initialize counter
    cntr = -1

    msap_sb = np.zeros(settings.N_f * settings.N_b)

    for k in np.arange(settings.N_f):

        # Extract msap_k
        msap_in_k = msap_in[k]

        # Normalize the mean-square acoustic pressure: divide the vector by the average value
        # Computational fix for equations for u / v
        if sum(msap_in) == 0:
            msap_proc = msap_in
        else:
            msap_proc = msap_in / (np.sum(msap_in) / msap_in.shape[0])

        if msap_proc[k] == 0:
                msap_sb[k * settings.N_b:(k + 1) * settings.N_b] = (np.sum(msap_in) ** 0) * np.zeros([settings.N_b])
        else:
            # Compute slope of spectrum
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 8-9
            if 0 < k < settings.N_f - 1:
                u = msap_proc[k] / msap_proc[k - 1]
                v = msap_proc[k + 1] / msap_proc[k]
            elif k == 0:
                u = msap_proc[1] / msap_proc[0]
                v = msap_proc[1] / msap_proc[0]
            elif k == settings.N_f - 1:
                u = msap_proc[k] / msap_proc[k - 1]
                v = msap_proc[k] / msap_proc[k - 1]

            # Compute constant A
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 12 + Berton ground-effects paper
            A = 1
            for h in np.arange(1, m + 1):
                A = A + u ** ((h - m - 1) / settings.N_b) + v ** ((h) / settings.N_b)

            # Compute MSAP in sub-bands
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 10 + Berton ground-effects paper
            for h in np.arange(settings.N_b):
                # Update counter
                cntr = cntr + 1

                # Compute subband msap
                if 0 <= h <= m - 1:
                    msap_sb[cntr] = (msap_in_k / A) * u ** ((h - m) / settings.N_b)
                elif h == m:
                    msap_sb[cntr] = (msap_in_k / A)
                elif m + 1 <= h <= settings.N_b - 1:
                    msap_sb[cntr] = (msap_in_k / A) * v ** ((h - m) / settings.N_b)

    return msap_sb
