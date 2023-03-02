import jax.numpy as jnp


def jet_shock_sae876(jet_V, jet_Tt, jet_A, jet_M, M_0, c_0, T_0, theta, f, settings, aircraft, tables):
        
    """
    Compute jet shock noise mean-square acoustic pressure (msap).
    """

    # Extract inputs
    jet_V_star = jet_V/c_0
    jet_A_star = jet_A/settings['A_e']
    jet_Tt_star = jet_Tt/T_0
    r_s_star = settings['r_0'] / jnp.sqrt(settings['A_e'])
    jet_delta = 0.

    # Calculate msap for all frequencies
    # If the jet is supersonic: shock cell noise
    if jet_M > 1.:
        # Calculate beta function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 4
        beta = (jet_M ** 2 - 1) ** 0.5

        # Calculate eta (exponent of the pressure ratio parameter)
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 5
        if beta > 1:
            if jet_Tt_star < 1.1:
                eta = 1.
            else:
                eta = 2.
        else:
            eta = 4.

        # Calculate f_star
        # Source: Zorumski report 1982 part 2. Chapter 8.5 page 8-5-1 (symbols)
        f_star = f * jnp.sqrt(settings['A_e']) / c_0

        # Calculate sigma parameter
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 3
        sigma = 7.80 * beta * (1 - M_0 * jnp.cos(jnp.pi / 180 * theta)) * jnp.sqrt(jet_A_star) * f_star
        log10sigma = jnp.log10(sigma)

        # Calculate W function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 6-7
        b = 0.23077
        W = 0
        for k in jnp.arange(1, settings['n_shock']):
            sum_inner = 0
            for m in jnp.arange(settings['n_shock'] - k):
                # Calculate q_km
                q_km = 1.70 * k / jet_V_star * (1 - 0.06 * (m + (k + 1) / 2)) * (1 + 0.7 * jet_V_star * jnp.cos(jnp.pi / 180 * theta))

                # Calculate inner sum (note: the factor b in the denominator below the sine should not be there: to get same graph as Figure 4)
                sum_inner = sum_inner + jnp.sin((b * sigma * q_km / 2)) / (sigma * q_km) * jnp.cos(sigma * q_km)

            # Compute the correlation coefficient spectrum C
            # Source: Zorumski report 1982 part 2. Chapter 8.5 Table II
            C = jnp.interp(log10sigma, tables.source.jet.C_log10sigma, tables.source.jet.C_data)

            # Add outer loop to the shock cell interference function
            W = W + (4. / (settings['n_shock'] * b))* sum_inner * C ** (k ** 2)

        # Calculate the H function
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Table III (+ linear extrapolation in logspace for log10sigma < 0; as given in SAEARP876)
        log10H = jnp.interp(log10sigma, tables.source.jet.H_log10sigma, tables.source.jet.H_data)

        # Source: Zorumski report 1982 part 2. Chapter 8.5.4
        if jet_Tt_star < 1.1:
            log10H = log10H - 0.2
        H = (10 ** log10H)

        # Calculate mean-square acoustic pressure (msap)
        # Multiply with number of engines and normalize msap by reference pressure
        # Source: Zorumski report 1982 part 2. Chapter 8.5 Equation 1
        msap = 1.92e-3 * jet_A_star / (4 * jnp.pi * r_s_star ** 2) * (1 + W) / (1 - M_0 * jnp.cos(jnp.pi / 180. * (theta - jet_delta))) ** 4 * beta ** eta * H * (aircraft.n_eng/settings['p_ref']**2)
    
    else:
        msap = jnp.zeros(settings['n_frequency_bands'],) * jet_M ** 0

    return msap


