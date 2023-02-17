import numpy as np
from typing import Tuple

def pnlt(levels, spl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute perceived noise level, tone corrected [PNdB]

    :param levels: pyNA component computing noise levels
    :type levels: Levels
    :param spl: sound pressure level [dB]
    :type spl: np.ndarray

    :return: pnlt, C
    :rtype: np.ndarray
    """
    # Load data
    settings = levels.options['settings']
    n_t = levels.options['n_t']

    # Initialize solution matrices
    pnl = np.zeros(n_t)
    pnlt = np.zeros(n_t)
    noy = np.zeros((n_t, settings['n_frequency_bands']))
    c_max = np.zeros(n_t)
    C = np.zeros((n_t, settings['n_frequency_bands']))

    for k in np.arange(n_t):
        spl_k = spl[k, :]

        # Compute noy
        # Source: ICAO Annex 16 Appendix 2 section 4.2 Step 1
        N = np.zeros(settings['n_frequency_bands'])
        for i in np.arange(settings['n_frequency_bands']):
            # Source: ICAO Annex 16 Appendix 2 Table A2-3 (Noy tables)
            spl_a = np.array([91. , 85.9, 87.3, 79. , 79.8, 76. , 74. , 74.9, 94.6,  1e8,  1e8, 1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8,  1e8, 44.3, 50.7])
            spl_b = np.array([64, 60, 56, 53, 51, 48, 46, 44, 42, 40, 40, 40, 40, 40, 38, 34, 32, 30, 29, 29, 30, 31, 34, 37])
            spl_c = np.array([52, 51, 49, 47, 46, 45, 43, 42, 41, 40, 40, 40, 40, 40, 38, 34, 32, 30, 29, 29, 30, 31, 34, 37])
            spl_d = np.array([49, 44, 39, 34, 30, 27, 24, 21, 18, 16, 16, 16, 16, 16, 15, 12,  9, 5,  4,  5,  6, 10, 17, 21])
            spl_e = np.array([55, 51, 46, 42, 39, 36, 33, 30, 27, 25, 25, 25, 25, 25, 23, 21, 18, 15, 14, 14, 15, 17, 23, 29])
            m_b   = np.array([0.043478, 0.04057 , 0.036831, 0.036831, 0.035336, 0.033333, 0.033333, 0.032051, 0.030675, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.02996 , 0.02996 , 0.02996 , 0.02996 , 0.02996 , 0.02996 , 0.02996 , 0.042285, 0.042285])
            m_c   = np.array([0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8,      1e8, 0.02996 , 0.02996 ])
            m_d   = np.array([0.07952 , 0.06816 , 0.06816 , 0.05964 , 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.05964 , 0.053013, 0.053013, 0.047712, 0.047712, 0.053013, 0.053013, 0.06816 , 0.07952 , 0.05964 ])
            m_e   = np.array([0.058098, 0.058098, 0.052288, 0.047534, 0.043573, 0.043573, 0.040221, 0.037349, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.040221, 0.037349, 0.034859, 0.034859, 0.034859, 0.034859, 0.037349, 0.037349, 0.043573])
            if spl_a[i] <= spl_k[i]:
                N[i] = 10 ** (m_c[i] * (spl_k[i] - spl_c[i]))
            elif spl_b[i] <= spl_k[i] <= spl_a[i]:
                N[i] = 10 ** (m_b[i] * (spl_k[i] - spl_b[i]))
            elif spl_e[i] <= spl_k[i] <= spl_b[i]:
                N[i] = 0.3 * 10 ** (m_e[i] * (spl_k[i] - spl_e[i]))
            elif spl_d[i] <= spl_k[i] <= spl_e[i]:
                N[i] = 0.1 * 10 ** (m_d[i] * (spl_k[i] - spl_d[i]))
            else:
                # Generate function N(SPL) that is always 0 for any SPL
                N[i] = spl_k[i] ** 0 - 1

        # Source: ICAO Annex 16 Appendix 2 section 4.2 Step 2
        noy[k, :] = N
        n_max = np.max(N)
        n_t = n_max + 0.15 * (np.sum(N) - n_max)

        # Compute perceived noise level (pnl)
        # Source: ICAO Annex 16 Appendix 2 section 4.2 Step 3
        if n_t < 1e-10:
            pnl[k] = n_t ** 0 - 1
        else:
            pnl[k] = 40 + 10. / np.log10(2) * np.log10(n_t)

        # Spectral irregularities correction
        # Step 1: Compute the slope of SPL
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 1
        s = np.zeros(settings['n_frequency_bands'])
        for i in np.arange(settings['n_frequency_bands']):
            # Up to the 3th band the table has no value
            if i <= 2:
                s[i] = np.nan
            # Start at the 4th band
            else:
                s[i] = spl_k[i] - spl_k[i - 1]

        # Step 2: Compute the absolute value of the slope and compare to 5
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 2
        slope = np.zeros(settings['n_frequency_bands'])
        slope_large = np.zeros(settings['n_frequency_bands'])
        for i in np.arange(settings['n_frequency_bands']):
            # Compute the absolute value of the slope
            slope[i] = s[i] - s[i - 1]
            # Check if slope is larger than 5
            if abs(slope[i]) > 5:
                slope_large[i] = 1

        # Step 3: Compute the encircled values of SPL
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 3
        spl_large = np.zeros(settings['n_frequency_bands'])
        for i in np.arange(settings['n_frequency_bands']):
            # Check if value of slope is encircled
            if slope_large[i] == 1:
                # Check if value of slope is positive and greater than previous slope
                if s[i] > 0 and s[i] > s[i - 1]:
                    spl_large[i] = 1
                elif s[i] <= 0 < s[i - 1]:
                    spl_large[i - 1] = 1

        # Step 4: Compute new adjusted sound pressure levels SPL'
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 4
        spl_p = np.zeros(settings['n_frequency_bands'])
        for i in np.arange(settings['n_frequency_bands']):
            if spl_large[i] == 0:
                spl_p[i] = spl_k[i]
            elif spl_large[i] == 1:
                if i <= 22:
                    spl_p[i] = 0.5 * (spl_k[i - 1] + spl_k[i + 1])
                elif i == 23:
                    spl_p[i] = spl_k[22] + s[22]

        # Step 5: Recompute the slope s'
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 5
        s_p = np.zeros(settings['n_frequency_bands'] + 1)
        for i in np.flip(np.arange(settings['n_frequency_bands']), 0):
            # From 4th band onwards
            if i > 2:
                s_p[i] = spl_p[i] - spl_p[i - 1]
            # At the 3rd band
            elif i == 2:
                s_p[i] = s_p[i + 1]
        # Compute 25th imaginary band
        s_p[24] = s_p[23]

        # Step 6: Compute arithmetic average of the 3 adjacent slopes
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 6
        s_bar = np.zeros(settings['n_frequency_bands'])
        for i in np.arange(2, settings['n_frequency_bands'] - 1):
            s_bar[i] = 1. / 3. * (s_p[i] + s_p[i + 1] + s_p[i + 2])

        # Step 7: Compute final 1/3 octave-band sound pressure level
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 7
        spl_pp = np.zeros(settings['n_frequency_bands'])
        for i in np.arange(2, settings['n_frequency_bands']):
            if i == 2:
                spl_pp[i] = spl_k[i]
            elif i > 2:
                spl_pp[i] = spl_pp[i - 1] + s_bar[i - 1]

        # Step 8: Compute the difference between SPL and SPL_pp
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 8
        F = np.zeros(settings['n_frequency_bands'])
        for i in np.arange(settings['n_frequency_bands']):
            # Compute the difference and limit at 1.5
            F[i] = spl_k[i] - spl_pp[i]
            # Check values larger than 1.5 (ICAO Appendix 2-16)
            if F[i] < 1.5:
                F[i] = 0

        # Step 9: Compute the correction factor C
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 9
        C_j = np.zeros(settings['n_frequency_bands'])
        for i in np.arange(2, settings['n_frequency_bands']):
            if i < 10:  # Frequency in [50,500[
                if 1.5 <= F[i] < 3:
                    C_j[i] = F[i] / 3. - 0.5
                elif 3. <= F[i] < 20.:
                    C_j[i] = F[i] / 6.
                elif F[i] >= 20.:
                    C_j[i] = 3. + 1 / 3.
            elif 10 <= i <= 20:  # Frequency in [500,5000]
                if 1.5 <= F[i] < 3.:
                    C_j[i] = 2. * F[i] / 3. - 1.0
                elif 3 <= F[i] < 20:
                    C_j[i] = F[i] / 3.
                elif F[i] >= 20:
                    C_j[i] = 6. + 2. / 3.
            elif i > 20:  # Frequency in ]5000,10000]
                if 1.5 <= F[i] < 3.:
                    C_j[i] = F[i] / 3. - 0.5
                elif 3. <= F[i] < 20.:
                    C_j[i] = F[i] / 6.
                elif F[i] >= 20.:
                    C_j[i] = 3. + 1. / 3.
        C[k, :] = C_j

        # Step 10: Compute the largest of the tone correction
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 10
        if not settings['tones_under_800Hz']:
            c_max[k] = np.max(C_j[13:])
        else:
            c_max[k] = np.max(C_j)

        # Compute tone-corrected perceived noise level (pnlt)
        # Source: ICAO Annex 16 Appendix 2 section 4.3 Step 10
        pnlt[k] = pnl[k] + c_max[k]

    return noy, pnl, pnlt, c_max, C
