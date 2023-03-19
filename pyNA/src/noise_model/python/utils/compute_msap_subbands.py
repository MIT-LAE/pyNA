import jax.numpy as jnp


def compute_msap_subbands(msap_in, settings):
    """
    Compute subfrequency bands for a given 1/3rd octave frequency spectrum.

    :param msap_in: mean-square acoustic pressure of the source (re. rho_0,^2c_0^2) [-]
    :type msap_in: 
    :param settings: pyna settings
    :type settings: 
    
    :return: mean-square acoustic pressure of the source, split into sub-frequency bands (re. rho_0,^2c_0^2) [-]
    :rtype: msap_out [settings['n_frequency_bands'] * settings['n_frequency_subbands'], ]

    """

    # Integer [-]
    m = (settings['n_frequency_subbands'] - 1) / 2

    # Initialize counter
    cntr = -1

    msap_sb = jnp.zeros(settings['n_frequency_bands'] * settings['n_frequency_subbands'], )
    for k in jnp.arange(settings['n_frequency_bands']):

        # Normalize the mean-square acoustic pressure: divide the vector by the average value
        # Computational fix for equations for u / v
        if sum(msap_in) == 0:
            msap_proc = msap_in
        else:
            msap_proc = msap_in / (jnp.sum(msap_in) / msap_in.shape[0])

        if msap_in[k] == 0:
            msap_sb[k * settings['n_frequency_subbands']:(k + 1) * settings['n_frequency_subbands']] = (jnp.sum(msap_in) ** 0) * jnp.zeros([settings['n_frequency_subbands']])
        else:
            # Compute slope of spectrum
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 8-9
            if 0 < k < settings['n_frequency_bands'] - 1:
                u = msap_proc[k] / msap_proc[k - 1]
                v = msap_proc[k + 1] / msap_proc[k]
            elif k == 0:
                u = msap_proc[1] / msap_proc[0]
                v = msap_proc[1] / msap_proc[0]
            elif k == settings['n_frequency_bands'] - 1:
                u = msap_proc[k] / msap_proc[k - 1]
                v = msap_proc[k] / msap_proc[k - 1]

            # Compute constant A
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 12 + Berton ground-effects paper
            A = 1
            for h in jnp.arange(1, m + 1):
                A = A + u ** ((h - m - 1) / settings['n_frequency_subbands']) + v ** ((h) / settings['n_frequency_subbands'])

            # Compute MSAP in sub-bands
            # Source: Zorumski report 1982 part 1. Chapter 5.1 Equation 10 + Berton ground-effects paper
            for h in jnp.arange(settings['n_frequency_subbands']):
                # Update counter
                cntr = cntr + 1

                # Compute subband msap
                if 0 <= h <= m - 1:
                    msap_sb = msap_sb.at[cntr].set((msap_in[k] / A) * u ** ((h - m) / settings['n_frequency_subbands']))
                elif h == m:
                    msap_sb = msap_sb.at[cntr].set((msap_in[k] / A))
                elif m + 1 <= h <= settings['n_frequency_subbands'] - 1:
                    msap_sb = msap_sb.at[cntr].set((msap_in[k] / A) * v ** ((h - m) / settings['n_frequency_subbands']))

    return msap_sb
