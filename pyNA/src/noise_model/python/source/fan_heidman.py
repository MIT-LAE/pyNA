import jax.numpy as jnp


def inlet_broadband(settings, theta, M_tip, tsqem, fan_M_d, fan_RSS, tables):
    """
    Compute the broadband component of the fan inlet mean-square acoustic pressure (msap).

    :param settings: pyna settings
    :type settings: Dict[str, Any]
    :param theta: polar directivity angle [deg]
    :type theta
    :param M_tip: relative (i.e., helical) tip Mach number [-]
    :type M_tip
    :param tsqem: broadband temperature-flow power base term [-]
    :type tsqem
    :param fan_M_d: fan rotor relative tip Mach number at design [-]
    :type fan_M_d
    :param fan_RSS: fan rotor-stator spacing [%]
    :type fan_RSS

    :return: bblv_I
    :rtype
    """

    # Fan inlet broadband noise component:
    if settings['fan_BB_method'] == 'original':
        # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A):
        if fan_M_d <= 1:
            if M_tip <= 0.9:
                f1ib = 58.5
            else:
                f1ib = 58.5 - 20 * jnp.log10(M_tip / 0.9)
        else:
            if M_tip <= 0.9:
                f1ib = 58.5 + 20 * jnp.log10(fan_M_d)
            else:
                f1ib = 58.5 + 20 * jnp.log10(fan_M_d) - 20 * jnp.log10(M_tip / 0.9)

        # Rotor-stator correction term (F2 of Eqn 4, Figure 6B):
        if not settings['fan_id']:
            f2ib = -5 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS <= 100:
                f2ib = -5 * jnp.log10(fan_RSS / 300)
            else:
                f2ib = -5 * jnp.log10(100 / 300)  # This is set to a constant 2.3856
    elif settings['fan_BB_method'] == 'allied_signal':
        # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A, modified by AlliedSignal):
        if fan_M_d <= 1:
            if M_tip <= 0.9:
                f1ib = 55.5
            else:
                f1ib = 55.5 - 20 * jnp.log10(M_tip / 0.9)
        else:
            if M_tip <= 0.9:
                f1ib = 55.5 + 20 * jnp.log10(fan_M_d)
            else:
                f1ib = 55.5 + 20 * jnp.log10(fan_M_d) - 20 * jnp.log10(M_tip / 0.9)

        # Rotor-stator spacing correction term (F2 of Eqn 4, Figure 6B):
        if not settings['fan_id']:
            f2ib = -5 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS <= 100:
                f2ib = -5 * jnp.log10(fan_RSS / 300)
            else:
                f2ib = -5 * jnp.log10(100. / 300)  # This is set to a constant 2.3856
    elif settings['fan_BB_method'] == 'geae':
        # Tip Mach-dependent term (F1 of Eqn 4 in report, Figure 4A, modified by GE):
        if fan_M_d <= 1:
            if M_tip <= 0.9:
                f1ib = 58.5
            else:
                f1ib = 58.5 - 50 * jnp.log10(M_tip / 0.9)
        else:
            if M_tip <= 0.9:
                f1ib = 58.5 + 20 * jnp.log10(fan_M_d)
            else:
                f1ib = 58.5 + 20 * jnp.log10(fan_M_d) - 50 * jnp.log10(M_tip / 0.9)

        # Rotor-stator spacing correction term (F2 of Eqn 4, Figure 6B):
        f2ib = 0
    elif settings['fan_BB_method'] == 'kresja':
        # Tip Mach-dependent term (F1, of Eqn 4 in report, Figure 4A, modified by Krejsa):
        if fan_M_d <= 1:
            if M_tip < 0.72:
                f1ib = 34 + 20 * jnp.log10(1. / 1.245)
            else:
                f1ib = 34 - 43 * (M_tip - 0.72) + 20 * jnp.log10(1. / 1.245)
        else:
            if M_tip < 0.72:
                f1ib = 34 + 20 * jnp.log10(fan_M_d / 1.245)
            else:
                f1ib = 34 - 43 * (M_tip - 0.72) + 20 * jnp.log10(fan_M_d/ 1.245)

        # Rotor-stator spacing correction term (F2, of Eqn 4, Figure 6B):
        if not settings['fan_id']:
            f2ib = -5 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS <= 100:
                f2ib = -5 * jnp.log10(fan_RSS / 300)
            else:
                f2ib = -5 * jnp.log10(100 / 300)  # This is set to a constant 2.3856
    else:
        raise ValueError('Invalid fan_BB_method specified. Specify: original / allied_signal / geae / kresja.')

    # Theta correction term (F3 of Eqn 4, Figure 7A):
    f3ib = jnp.interp(theta, tables.source.fan.f3ib_theta, tables.source.fan.f3ib_data)

    # Component value:
    bblv_I = tsqem + f1ib + f2ib + f3ib

    return bblv_I

def discharge_broadband(settings, theta, M_tip, tsqem, fan_M_d, fan_RSS, tables):
    """
    Compute the broadband component of the fan discharge mean-square acoustic pressure (msap).

    :param settings: pyna settings
    :type settings: Dict[str, Any]
    :param theta: polar directivity angle [deg]
    :type theta
    :param M_tip: relative (i.e., helical) tip Mach number [-]
    :type M_tip
    :param tsqem: broadband temperature-flow power base term [-]
    :type tsqem
    :param fan_M_d: fan rotor relative tip Mach number at design [-]
    :type fan_M_d
    :param fan_RSS: fan rotor-stator spacing [%]
    :type fan_RSS

    :return: bblv_D
    :rtype
    """
    # Fan discharge broadband noise component
    if settings['fan_BB_method'] == 'original':
        # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B):
        if fan_M_d <= 1:
            if M_tip <= 1:
                f1db = 60
            else:
                f1db = 60 - 20 * jnp.log10(M_tip / 1)
        else:
            if M_tip <= 1:
                f1db = 60 + 20 * jnp.log10(fan_M_d)
            else:
                f1db = 60 + 20 * jnp.log10(fan_M_d) - 20 * jnp.log10(M_tip / 1)

        # Rotor-stator correction term (F2 of Eqn 4, Figure 6B):
        if not settings['fan_id']:
            f2db = -5 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS <= 100:
                f2db = -5 * jnp.log10(fan_RSS / 300)
            else:
                f2db = -5 * jnp.log10(100 / 300)  # This is set to a constant 2.3856
    elif settings['fan_BB_method'] == 'allied_signal':
        # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B, modified by AlliedSignal):
        if fan_M_d <= 1:
            if M_tip <= 1:
                f1db = 58
            else:
                f1db = 58 - 20 * jnp.log10(M_tip / 1)
        else:
            if M_tip <= 1:
                f1db = 58 + 20 * jnp.log10(fan_M_d)
            else:
                f1db = 58 + 20 * jnp.log10(fan_M_d) - 20 * jnp.log10(M_tip / 1)

        # Rotor-stator spacing correction term (F2 of Eqn 10, Figure 6B, modified by AlliedSignal):
        if not settings['fan_id']:
            f2db = -5 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS <= 100:
                f2db = -5 * jnp.log10(fan_RSS / 300)
            else:
                f2db = -5 * jnp.log10(100 / 300)  # This is set to a constant 2.3856
    elif settings['fan_BB_method'] == 'geae':
        # Tip Mach-dependent term (F1 of Eqn 10 in report, Figure 4B, modified by GE):
        if fan_M_d <= 1:
            if M_tip <= 1:
                f1db = 63
            else:
                f1db = 63 - 30 * jnp.log10(M_tip / 1)
        else:
            if M_tip <= 1:
                f1db = 63 + 20 * jnp.log10(fan_M_d)
            else:
                f1db = 63 + 20 * jnp.log10(fan_M_d) - 30 * jnp.log10(M_tip / 1)

        # Rotor-stator spacing correction term (F2 of Eqn 10, Figure 6B, modified by GE):
        f2db = -5 * jnp.log10(fan_RSS / 300)
    elif settings['fan_BB_method'] == 'kresja':
        # Tip Mach-dependent term (F1, of Eqn 10 in report, Figure 4B, modified by Krejsa):
        if fan_M_d <= 1:
            # If M_tip < 0.65 Then
            #   f1dbBkrejsa = 34 + 20 * jnp.log10(1 / 1.245)
            # Else
            f1db = 34 - 17 * (M_tip - 0.65) + 20 * jnp.log10(1 / 1.245)
        else:
            # If M_tip < 0.65 Then
            #   f1dbBkrejsa = 34 + 20 * jnp.log10(fan_M_d / 1.245)
            # Else
            f1db = 34 - 17 * (M_tip - 0.65) + 20 * jnp.log10(fan_M_d / 1.245)

        # Rotor-stator spacing correction term (F2, of Eqn 10, Figure 6B):
        if not settings['fan_id']:
            f2db = -5 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS <= 100:
                f2db = -5 * jnp.log10(fan_RSS / 300)
            else:
                f2db = -5 * jnp.log10(100 / 300)  # This is set to a constant 2.3856
    else:
        raise ValueError('Invalid fan_BB_method specified. Specify: original / allied_signal / geae / kresja.')

    # Theta correction term (F3 of Eqn 10, Figure 7B):
    f3db = jnp.interp(theta, tables.source.fan.f3db_theta, tables.source.fan.f3db_data)

    # Added noise factor if there are inlet guide vanes present:
    if settings['fan_igv']:
        cdb = 3
    else:
        cdb = 0

    # Component value:
    bblv_D = tsqem + f1db + f2db + f3db + cdb

    return bblv_D

def inlet_tones(settings, theta, M_tip, tsqem, fan_M_d, fan_RSS, tables):
    """
    Compute the tone component of the fan inlet mean-square acoustic pressure (msap)

    :param settings: pyna settings
    :type settings: Dict[str, Any]
    :param theta: polar directivity angle [deg]
    :type theta
    :param M_tip: relative (i.e., helical) tip Mach number [-]
    :type M_tip
    :param tsqem: tone temperature-flow power base term [-]
    :type tsqem
    :param fan_M_d: fan rotor relative tip Mach number at design [-]
    :type fan_M_d
    :param fan_RSS: fan rotor-stator spacing [%]
    :type fan_M_d

    :return: tonlv_I
    :rtype
    """

    # Fan inlet discrete tone noise component:

    if settings['fan_RS_method'] == 'original':
        # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A):
        if fan_M_d <= 1:
            if M_tip <= 0.72:
                f1ti = 60.5
            else:
                f1ti_a = 60.5 + 50 * jnp.log10(M_tip / 0.72)
                f1ti_b = 59.5 + 80 * jnp.log10(1. / M_tip)
                if f1ti_a < f1ti_b:
                    f1ti = f1ti_a
                else:
                    f1ti = f1ti_b
        else:
            if M_tip <= 0.72:
                f1ti = 60.5 + 20 * jnp.log10(fan_M_d)
            else:
                f1ti_a = 60.5 + 20 * jnp.log10(fan_M_d) + 50 * jnp.log10(M_tip / 0.72)
                # Note the 1975 version of NASA TMX-71763 writes it this way:
                # f1ti_b = 59.5 + 20 * jnp.log10(fan_M_d) + 80 * jnp.log10(fan_M_d / M_tip)
                # But the 1979 version of NASA TMX-71763 writes it this way:
                f1ti_b = 59.5 + 80 * jnp.log10(fan_M_d / M_tip)
                if f1ti_a < f1ti_b:
                    f1ti = f1ti_a
                else:
                    f1ti = f1ti_b

        # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12):
        if not settings['fan_id']:
            f2ti = -10 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS < 100:
                f2ti = -10 * jnp.log10(fan_RSS / 300)
            else:
                f2ti = -10 * jnp.log10(100 / 300)  # This is set to a constant 4.7712
    elif settings['fan_RS_method'] == 'allied_signal':
        # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A, modified by AlliedSignal):
        if fan_M_d <= 1:
            if M_tip <= 0.72:
                f1ti = 54.5
            else:
                f1ti_a = 54.5 + 50 * jnp.log10(M_tip / 0.72)
                f1ti_b = 53.5 + 80 * jnp.log10(1. / M_tip)
                if f1ti_a < f1ti_b:
                    f1ti = f1ti_a
                else:
                    f1ti = f1ti_b
        else:
            if M_tip <= 0.72:
                f1ti = 54.5 + 20 * jnp.log10(fan_M_d)
            else:
                f1ti_a = 54.5 + 20 * jnp.log10(fan_M_d) + 50 * jnp.log10(M_tip / 0.72)
                f1ti_b = 53.5 + 80 * jnp.log10(fan_M_d / M_tip)
                if f1ti_a < f1ti_b:
                    f1ti = f1ti_a
                else:
                    f1ti = f1ti_b

        # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12, modified by AlliedSignal):
        if not settings['fan_id']:
            f2ti = -10 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS < 100:
                f2ti = -10 * jnp.log10(fan_RSS / 300)
            else:
                f2ti = -10 * jnp.log10(100./ 300)  # This is set to a constant 4.7712
    elif settings['fan_RS_method'] == 'geae':
        # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A, modified by GE):
        if fan_M_d <= 1:
            if M_tip <= 0.72:
                f1ti = 60.5
            else:
                f1ti_a = 60.5 + 50 * jnp.log10(M_tip / 0.72)
                f1ti_b = 64.5 + 80 * jnp.log10(1. / M_tip)
                if f1ti_a < f1ti_b:
                    f1ti = f1ti_a
                else:
                    f1ti = f1ti_b
        else:
            if M_tip <= 0.72:
                f1ti = 60.5 + 20 * jnp.log10(fan_M_d)
            else:
                f1ti_a = 60.5 + 20 * jnp.log10(fan_M_d) + 50 * jnp.log10(M_tip / 0.72)
                f1ti_b = 64.5 + 80 * jnp.log10(fan_M_d) - 80 * jnp.log10(M_tip)
                if f1ti_a < f1ti_b:
                    f1ti = f1ti_a
                else:
                    f1ti = f1ti_b

        # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12, modified to zero by GE):
        f2ti = 0
    elif settings['fan_RS_method'] == 'kresja':
        # Tip Mach-dependent term (F1 of Eqn 6 in report, Figure 10A, modified by Krejsa):
        if fan_M_d <= 1:
            f1ti = 42 - 20 * M_tip + 20 * jnp.log10(1. / 1.245)
        else:
            f1ti = 42 - 20 * M_tip + 20 * jnp.log10(fan_M_d / 1.245)

        # Rotor-stator spacing correction term (F2 of Eqn 6, Figure 12):
        if not settings['fan_id']:
            f2ti = -10 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS < 100:
                f2ti = -10 * jnp.log10(fan_RSS / 300)
            else:
                f2ti = -10 * jnp.log10(100 / 300)  # This is set to a constant 4.7712
    else:
        raise ValueError('Invalid fan_BB_method specified. Specify: original / allied_signal / geae / kresja.')

    # Theta correction term (F3 of Eqn 6, Figure 13A):
    f3ti = jnp.interp(theta, tables.source.fan.f3ti_theta, tables.source.fan.f3ti_data)

    # Component value:
    tonlv_I = tsqem + f1ti + f2ti + f3ti

    return tonlv_I

def discharge_tones(settings, theta, M_tip, tsqem, fan_M_d, fan_RSS, tables):
    """
    Compute the tone component of the fan discharge mean-square acoustic pressure (msap)

    :param settings: pyna settings
    :type settings: Dict[str, Any]
    :param theta: polar directivity angle [deg]
    :type theta
    :param M_tip: relative (i.e., helical) tip Mach number [-]
    :type M_tip
    :param tsqem: broadband temperature-flow power base term [-]
    :type tsqem
    :param fan_M_d: fan rotor relative tip Mach number at design [-]
    :type fan_M_d
    :param fan_RSS: fan rotor-stator spacing [%]
    :type fan_M_d

    :return: tonlv_X
    :rtype
    """

    # Fan discharge discrete tone noise component:

    if settings['fan_RS_method'] == 'original':
        # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B):
        if fan_M_d <= 1:
            if M_tip <= 1:
                f1td = 63
            else:
                f1td = 63 - 20 * jnp.log10(M_tip / 1)
        else:
            if M_tip <= 1:
                f1td = 63 + 20 * jnp.log10(fan_M_d)
            else:
                f1td = 63 + 20 * jnp.log10(fan_M_d) - 20 * jnp.log10(M_tip / 1)

        # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12):
        if not settings['fan_id']:
            f2td = -10 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS < 100:
                f2td = -10 * jnp.log10(fan_RSS / 300)
            else:
                f2td = -10 * jnp.log10(100 / 300)  # This is set to a constant 4.7712
    elif settings['fan_RS_method'] == 'allied_signal':
        # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by AlliedSignal):
        if fan_M_d <= 1:
            if M_tip <= 1:
                f1td = 59
            else:
                f1td = 59 - 20 * jnp.log10(M_tip / 1)
        else:
            if M_tip <= 1:
                f1td = 59 + 20 * jnp.log10(fan_M_d)
            else:
                f1td = 59 + 20 * jnp.log10(fan_M_d) - 20 * jnp.log10(M_tip / 1)

        # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12, modified by AlliedSignal):
        if not settings['fan_id']:
            f2td = -10 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS < 100:
                f2td = -10 * jnp.log10(fan_RSS / 300)
            else:
                f2td = -10 * jnp.log10(100 / 300)  # This is set to a constant 4.7712
    elif settings['fan_RS_method'] == 'geae':
        # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by GE):
        if fan_M_d <= 1:
            if M_tip <= 1:
                f1td = 63
            else:
                f1td = 63 - 20 * jnp.log10(M_tip / 1)
        else:
            if M_tip <= 1:
                f1td = 63 + 20 * jnp.log10(fan_M_d)
            else:
                f1td = 63 + 20 * jnp.log10(fan_M_d) - 20 * jnp.log10(M_tip / 1)

        # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12, modified by GE):
        f2td = -10 * jnp.log10(fan_RSS / 300)
    elif settings['fan_RS_method'] == 'kresja':
        # Tip Mach-dependent term (F1 of Eqn 12 in report, Figure 10B, modified by Krejsa):
        if fan_M_d <= 1:
            f1td = 46 - 20 * M_tip + 20 * jnp.log10(1 / 1.245)
        else:
            f1td = 46 - 20 * M_tip + 20 * jnp.log10(fan_M_d / 1.245)

        # Rotor-stator spacing correction term (F2 of Eqn 12, Figure 12, modified by Krejsa):
        if not settings['fan_id']:
            f2td = -10 * jnp.log10(fan_RSS / 300)  # If no distortion
        else:
            if fan_RSS < 100:
                f2td = -10 * jnp.log10(fan_RSS / 300)
            else:
                f2td = -10 * jnp.log10(100 / 300)  # This is set to a constant 4.7712
    else:
        raise ValueError('Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja.')

    # Theta correction term (F3 of Eqn 6, Figure 13B):
    f3td = jnp.interp(theta, tables.source.fan.f3td_theta, tables.source.fan.f3td_data)

    # Added noise factor if there are inlet guide vanes:
    if settings['fan_igv']:
        cdt = 6
    else:
        cdt = 0

    # Component value:
    tonlv_X = tsqem + f1td + f2td + f3td + cdt

    return tonlv_X

def combination_tones(settings, freq, theta, M_tip, bpf, tsqem, tables):
    """
    Compute the combination tone component of the fan mean-square acoustic pressure (msap).

    :param settings: pyna settings
    :type settings: Dict[str, Any]
    :param freq: 1/3rd octave frequency bands [Hz]
    :type freq
    :param theta: polar directivity angle [deg]
    :type theta
    :param M_tip: relative (i.e., helical) tip Mach number [-]
    :type M_tip
    :param bpf: blade pass frequency
    :type bpf
    :param tsqem: tone temperature-flow power base term [-]
    :type tsqem

    :return: dcp
    :rtype
    """
    # Combination tone (multiple pure tone or buzzsaw) calculations:
    # Note the original Heidmann reference states that MPTs should be computed if
    # the tangential tip speed is supersonic, but the ANOPP implementation states MPTs
    # should be computed if the relative tip speed is supersonic.  The ANOPP implementation
    # is used here, i.e., if M_tip >= 1.0, MPTs are computed.

    # Initialize solution matrices
    dcp = jnp.zeros(settings['n_frequency_bands'])

    if M_tip >= 1:

        # Theta correction term (F2 of Eqn 8, Figure 16):
        f2ct = jnp.interp(theta, tables.source.fan.f2ct_theta, tables.source.fan.f2ct_data)

        if settings['fan_RS_method'] == 'original':
            sl3 = jnp.array([0, -30, -50, -30])
            sl4 = jnp.array([0, 30, 50, 50])
            yint3 = jnp.array([0, -9.0309, -30.103, -27.0927])
            yint4 = jnp.array([0, 9.0309, 30.103, 45.1545])

        elif settings['fan_RS_method'] == 'allied_signal':
            sl3 = jnp.array([0, -15, -50, -30])
            sl4 = jnp.array([0, 30, 50, 50])
            yint3 = jnp.array([0, -4.51545, -30.103, -27.0927])
            yint4 = jnp.array([0, 9.0309, 30.103, 45.1545])

        elif settings['fan_RS_method'] == 'geae':
            sl3 = jnp.array([0, -30, -50, -30])
            sl4 = jnp.array([0, 30, 50, 50])
            yint3 = jnp.array([0, -9.0309, -30.103, -27.0927])
            yint4 = jnp.array([0, 9.0309, 30.103, 45.1545])

        elif settings['fan_RS_method'] == 'kresja':
            sl3 = jnp.array([0, -20, -30, -20])
            sl4 = jnp.array([0, 20, 30, 30])
            yint3 = jnp.array([0, -6.0206, -18.0618, -18.0618])
            yint4 = jnp.array([0, 6.0206, 18.0618, 27.0927])

        # Noise adjustment (reduction) if there are inlet guide vanes, for all methods:
        if settings['fan_igv']:
            cct = -5
        else:
            cct = 0

        # Loop through the three sub-bpf terms:
        # K = 1; 1/2 bpf term
        # K = 2; 1/4 bpf term
        # K = 3; 1/8 bpf term
        for k in jnp.arange(1, 4):
            # Tip Mach-dependent term (F1 of Eqn 8 in Heidmann report, Figure 15A):
            if settings['fan_RS_method'] == 'original':
                # Original tip Mach number-dependent term of multiple pure tone noise
                fig15mt = jnp.array([0, 1.14, 1.25, 1.61])
                sl1 = jnp.array([0, 785.68, 391.81, 199.2])
                sl2 = jnp.array([0, -49.62, -50.06, -49.89])
                yint1 = jnp.array([0, 30, 30, 30])
                yint2 = jnp.array([0, 79.44, 83.57, 81.52])

                if M_tip < fig15mt[k]:
                    f1ct = sl1[k] * jnp.log10(M_tip) + yint1[k]
                else:
                    f1ct = sl2[k] * jnp.log10(M_tip) + yint2[k]
            elif settings['fan_RS_method'] == 'allied_signal':
                # Revised by AlliedSignal:  tip Mach number-dependent term of multiple pure tone noise.
                # Note that a 20log10 independent variable distribution is specified.
                fig15mt = jnp.array([0, 1.135, 1.135, 1.413])
                sl1 = jnp.array([0, 5.9036, 5.54769, 2.43439])
                sl2 = jnp.array([0, -0.632839, -0.632839, -1.030931])
                yint1 = jnp.array([0, 50, 50, 40])
                yint2 = jnp.array([0, 57.1896, 56.7981, 50.4058])

                if M_tip < fig15mt[k]:
                    f1ct = sl1[k] * 20 * jnp.log10(M_tip) + yint1[k]
                else:
                    f1ct = sl2[k] * 20 * jnp.log10(M_tip) + yint2[k]
            elif settings['fan_RS_method'] == 'geae':
                # Revised by GE: tip Mach number-dependent term of multiple pure tone noise
                fig15mt = jnp.array([0, 1.14, 1.25, 1.61])
                sl1 = jnp.array([0, 746.8608, 398.3077, 118.9406])
                sl2 = jnp.array([0, -278.96, -284.64, -43.52])
                yint1 = jnp.array([0, 30, 30, 36])
                yint2 = jnp.array([0, 88.37, 96.18, 69.6])

                if M_tip < fig15mt[k]:
                    f1ct = sl1[k] * jnp.log10(M_tip) + yint1[k]
                else:
                    f1ct = sl2[k] * jnp.log10(M_tip) + yint2[k]
            elif settings['fan_RS_method'] == 'kresja':
                fig15mt = jnp.array([0, 1.146, 1.322, 1.61])
                sl1 = jnp.array([0, 785.68, 391.81, 199.2])
                sl2 = jnp.array([0, -49.62, -50.06, -49.89])
                yint1 = jnp.array([0, -18, -15, -12])
                yint2 = jnp.array([0, 31.44, 38.57, 39.52])

                if M_tip < fig15mt[k]:
                    f1ct = sl1[k] * jnp.log10(M_tip) + yint1[k]
                else:
                    f1ct = sl2[k] * jnp.log10(M_tip) + yint2[k]
            else:
                raise ValueError('Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja.')

            ctlc = tsqem + f1ct + f2ct + cct
            # Frequency-dependent term (F3 of Eqn 9, Figure 14):
            fk = 2 ** K

            # Cycle through frequencies and make assignments:
            for j in jnp.arange(settings['n_frequency_bands']):
                fqfb = freq[j] / bpf

                if fqfb <= 1 / fk:
                    # For frequencies less than the subharmonic:
                    f3ct = sl4[k] * jnp.log10(fqfb) + yint4[k]
                else:
                    # For frequencies greater than the subharmonic:
                    f3ct = sl3[k] * jnp.log10(fqfb) + yint3[k]

                # Be sure to add the three sub-bpf components together at each frequency:
                dcp[j] = dcp[j] + 10 ** (0.1 * (ctlc + f3ct))

    return dcp

def calculate_cutoff(M_tip_tan, fan_B, fan_V):
    """
    Compute if the fan is in cut-off condition (0/1).

    :param M_tip_tan: tangential (i.e., radius*omega) tip Mach number [-]
    :type M_tip_tan
    :param fan_B: fan blade number [-]
    :type fan_B
    :param fan_V: fan vane number [-]
    :type fan_V

    :return: i_cut
    :rtype
    """

    # Vane/blade ratio parameter:
    vane_blade_ratio = 1 - fan_V / fan_B
    if vane_blade_ratio == 0:
        vane_blade_ratio = 1e-6

    # Fundamental tone cutoff parameter:
    # Source: Zorumski report 1982 part 2. Chapter 8.1 Eq. 8
    delta_cutoff = abs(M_tip_tan / vane_blade_ratio)
    # if the cutoff parameter is less than 1.05 and the tip Mach is less than unity, the fan is cut off
    # and fan noise does not propagate (i.e., the tones are reduced in magnitude):
    if delta_cutoff < 1.05:
        # Fan cut-off
        if M_tip_tan < 1:
            i_cut = delta_cutoff ** 0
        # No cutoff: supersonic tip mach number
        else:
            i_cut = delta_cutoff ** 0 - 1
    else:
        # No cutoff: poor choice of blades and vanes
        if M_tip_tan < 1:
            i_cut = delta_cutoff ** 0 - 1
        # No cutoff: supersonic tip mach number
        else:
            i_cut = delta_cutoff ** 0 - 1

    return i_cut

def calculate_harmonics(settings, freq, theta, tonlv_I, tonlv_X, i_cut, M_tip, bpf, comp):
    """
    Compute fan tone harmonics for inlet (dp) and discharge (dpx).

    :param settings: pyna settings
    :type settings: Dict[str, Any]
    :param freq: 1/3rd octave frequency bands [Hz]
    :type freq
    :param theta: polar directivity angle [deg]
    :type theta
    :param tonlv_I: inlet tone level [-]
    :type tonlv_I
    :param tonlv_X: discharge tone level [-]
    :type tonlv_X
    :param i_cut: cut-off parameter (0/1)
    :type i_cut
    :param M_tip: relative (i.e., helical) tip Mach number [-]
    :type M_tip
    :param bpf: blade pass frequency
    :param comp: fan component (fan_inlet / fan_discharge)
    :type comp: str

    :return: dp, dpx
    :rtype: jnp.ndarray

    """

    # Assign discrete interaction tones at bpf and harmonics to proper bins (see figures 8 and 9):
    # Initialize solution matrices
    dp = jnp.zeros(settings['n_frequency_bands'])
    dpx = jnp.zeros(settings['n_frequency_bands'])

    nfi = 1
    for ih in jnp.arange(1, settings['n_harmonics'] + 1):

        # Determine the tone fall-off rates per harmonic (harm_i and harm_x):
        if settings['fan_RS_method'] == 'original':
            if not settings['fan_igv']:
                # For fans without inlet guide vanes:
                if i_cut == 0:
                    # For cut-on fans, fundamental:
                    if ih == 1:
                        harm_i = 0
                        harm_x = 0
                    else:
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih - 1)
                        harm_x = 3 * (ih - 1)
                elif i_cut == 1:
                    # For cut-off fans, fundamental:
                    if ih == 1:
                        harm_i = 8
                        harm_x = 8
                    else:
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih - 1)
                        harm_x = 3 * (ih - 1)
                else:
                    raise ValueError('Cut-off value out of bounds.')
            else:
                # For fans with inlet guide vanes:
                if i_cut == 0:
                    # For cut-on fans, fundamental:
                    if ih == 1:
                        harm_i = 0
                        harm_x = 0
                    else:
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih + 1)
                        harm_x = 3 * (ih + 1)
                elif i_cut == 1:
                    # For cut-off fans, fundamental:
                    if ih == 1:
                        harm_i = 8
                        harm_x = 8
                    else:
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih + 1)
                        harm_x = 3 * (ih + 1)
                else:
                    raise ValueError('Cut-off value out of bounds.')
        elif settings['fan_RS_method'] == 'allied_signal':
            if not settings['fan_igv']:
                # For fans without inlet guide vanes:
                if i_cut == 0:
                    # For cut-on fans, fundamental:
                    if ih == 1:
                        harm_i = 0
                        harm_x = 0
                    elif ih == 2:
                        # For cut-on fans, second harmonic:
                        harm_i = 9.2
                        harm_x = 9.2
                    else:
                        # For cut-on fans, upper harmonics:
                        harm_i = 3 * ih + 1.8
                        harm_x = 3 * ih + 1.8
                elif i_cut == 1:
                    # For cut-off fans, fundamental:
                    if ih == 1:
                        harm_i = 8
                        harm_x = 8
                    elif ih == 2:
                        # For cut-off fans, second harmonic:
                        harm_i = 9.2
                        harm_x = 9.2
                    else:
                        # For cut-off fans, upper harmonics:
                        harm_i = 3 * ih + 1.8
                        harm_x = 3 * ih + 1.8
                else:
                    raise ValueError('Cut-off value out of bounds.')
            else:
                # For fans with inlet guide vanes:
                if i_cut == 0:
                    # For cut-on fans, fundamental:
                    if ih == 1:
                        harm_i = 0
                        harm_x = 0
                    else:
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih + 1)
                        harm_x = 3 * (ih + 1)
                elif i_cut == 1:
                    # For cut-off fans, fundamental:
                    if ih == 1:
                        harm_i = 8
                        harm_x = 8
                    else:
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih + 1)
                        harm_x = 3 * (ih + 1)
                else:
                    raise ValueError('Cut-off value out of bounds.')
        elif settings['fan_RS_method'] == 'geae':
            if not settings['fan_igv']:
                # For fans without inlet guide vanes:
                if i_cut == 0:
                    # For cut-on fans, fundamental:
                    if ih == 1:
                        harm_i = 0
                        harm_x = 0
                    else:
                        # For cut-on fans, harmonics:
                        if M_tip < 1.15:
                            harm_i = 6 * (ih - 1)
                            harm_x = 3 * (ih - 1)
                        else:
                            harm_i = 9 * (ih - 1)
                            harm_x = 3 * (ih - 1)
                elif i_cut == 1:
                    # For cut-off fans, fundamental:
                    if ih == 1:
                        harm_i = 8
                        harm_x = 8
                    else:
                        # For cut-off fans, harmonics:
                        if M_tip < 1.15:
                            harm_i = 6 * (ih - 1)
                            harm_x = 3 * (ih - 1)
                        else:
                            harm_i = 9 * (ih - 1)
                            harm_x = 3 * (ih - 1)
                else:
                    raise ValueError('Cut-off value out of bounds.')
            else:
                # For fans with inlet guide vanes:
                if i_cut == 0:
                    # For cut-on fans, fundamental:
                    if ih == 1:
                        harm_i = 0
                        harm_x = 0
                    else:
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih + 1)
                        harm_x = 3 * (ih + 1)
                elif i_cut == 1:
                    # For cut-off fans, fundamental:
                    if ih == 1:
                        harm_i = 8
                        harm_x = 8
                    else:
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih + 1)
                        harm_x = 3 * (ih + 1)
                else:
                    raise ValueError('Cut-off value out of bounds.')
        elif settings['fan_RS_method'] == 'kresja':
            if not settings['fan_igv']:
                # For fans without inlet guide vanes:
                if i_cut == 0:
                    # For cut-on fans, fundamental:
                    if ih == 1:
                        harm_i = 0
                        harm_x = 0
                    else:
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih - 1)
                        harm_x = 3 * (ih - 1)
                elif i_cut == 1:
                    # For cut-off fans, fundamental:
                    if ih == 1:
                        harm_i = 8
                        harm_x = 8
                    else:
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih - 1)
                        harm_x = 3 * (ih - 1)
                else:
                    raise ValueError('Cut-off value out of bounds.')
            else:
                # For fans with inlet guide vanes:
                if i_cut == 0:
                    # For cut-on fans, fundamental:
                    if ih == 1:
                        harm_i = 0
                        harm_x = 0
                    else:
                        # For cut-on fans, harmonics:
                        harm_i = 3 * (ih + 1)
                        harm_x = 3 * (ih + 1)
                elif i_cut == 1:
                    # For cut-off fans, fundamental:
                    if ih == 1:
                        harm_i = 8
                        harm_x = 8
                    else:
                        # For cut-off fans, harmonics:
                        harm_i = 3 * (ih + 1)
                        harm_x = 3 * (ih + 1)
                else:
                    raise ValueError('Cut-off value out of bounds.')
        else:
            raise ValueError('Invalid fan_RS_method specified. Specify: original / allied_signal / geae / kresja.')

        # Calculate TCS and distor
        if settings['fan_id'] and settings['fan_RS_method'] != 'geae':
            # Assign the increment to the fundamental tone along with a 10 dB per harmonic order fall-off
            # for cases with inlet flow distortion (see figure 9):
            distor = 10 ** (0.1 * tonlv_I - ih + 1)
            TCS = 0
        elif settings['fan_id'] and settings['fan_RS_method'] == 'geae':
            # Compute suppression factors for GE#s "Flight cleanup Turbulent Control Structure."
            # Approach or takeoff values to be applied to inlet discrete interaction tones
            # at bpf and 2bpf.  Accounts for observed in-flight tendencies.
            TCSTHA = 
            TCSAT1 = 
            TCSAT2 = 
            TCSAA1 = 
            TCSAA2 = 

            if settings['fan_ge_flight_cleanup'] == 'takeoff':
                if ih == 1:
                    TCS = jnp.interp(theta, tables.source.fan.fe_theta, tables.source.fan.fe_takeoff_1_data)
                elif ih == 2:
                    TCS = jnp.interp(theta, tables.source.fan.fe_theta, tables.source.fan.fe_takeoff_2_data)
                else:
                    TCS = 0.

            elif settings['fan_ge_flight_cleanup'] == 'approach':
                if ih == 1:
                    TCS = jnp.interp(theta, tables.source.fan.fe_theta, tables.source.fan.fe_approach_1_data)
                elif ih == 2:
                    TCS = jnp.interp(theta, tables.source.fan.fe_theta, tables.source.fan.fe_approach_2_data)
                else:
                    TCS = 0.

            elif settings['fan_ge_flight_cleanup'] == 'none':
                # Apply zero values (i.e., fan inlet flow is distorted):
                TCS = 0.

            # Inlet distortion effects are always included in basic inlet tone model of the GE method.
            # Flight cleanup levels are then subtracted from the inlet tones if the flow is not distorted.
            # The flight cleanup levels are set to zero if the flow is distorted.
            # Use the same increment as the original method and the same 10 dB per harmonic fall-off rate.
            distor = 10 ** (0.1 * (tonlv_I - TCS) - ih + 1)
        else:
            distor = 0
            TCS = 0

        # Calculate tone power
        if comp == 'fan_inlet':# or comp == 'inlet RS':
            tojnpwr_i = 10 ** (0.1 * (tonlv_I - harm_i - TCS)) + distor
        else:
            tojnpwr_i = 0

        if comp == 'fan_discharge':# or comp == 'discharge RS' or comp == 'total':
            tojnpwr_x = 10 ** (0.1 * (tonlv_X - harm_x))
        else:
            tojnpwr_x = 0

        # Compute filter bandwidths:
        filbw = 1  # Fraction of filter bandwidth with gain of unity (default to unity)
        F1 = 0.78250188 + 0.10874906 * filbw
        F2 = 1 - 0.10874906 * filbw
        F3 = 1 + 0.12201845 * filbw
        F4 = 1.2440369 - 0.12201845 * filbw

        # Cycle through frequencies and assign tones to 1/3rd octave bins:
        for l in jnp.arange(nfi - 1, settings['n_frequency_bands']):
            Frat = bpf * ih / freq[l]
            FR = 1
            if Frat < F1:
                break
            elif Frat > F4:
                ll = l
                continue
            elif Frat > F3:
                FR = (F4 - Frat) / (F4 - F3)
            elif Frat < F2:
                FR = (Frat - F1) / (F2 - F1)
            dp[l] = dp[l] + tojnpwr_i * FR
            dpx[l] = dpx[l] + tojnpwr_x * FR
            nfi = ll
            continue

    return dp, dpx


def fan_heidman(fan_DTt, fan_mdot, fan_N, M_0, c_0, rho_0, theta, f, settings, aircraft, tables, comp):

    """
    Calculates fan noise mean-square acoustic pressure (msap) using Berton's implementation of the fan noise method.

    :param comp: fan component (fan_inlet / fan_discharge)
    :type comp: str

    :return: msap_fan
    :rtype: jjnp.ndarray
    """
    
    ### Extract the ijnputs
    M_tip_tan = (aircraft.engine.fan_d / 2) * fan_N * 2 * jjnp.pi / 60 / c_0  # Tangential (i.e., radius*omega) tip Mach number: ! Doesn't need correction
    bpf = fan_N * aircraft.engine.fan_B / 60. / (1 - M_0 * jjnp.cos(theta * jjnp.pi / 180))  # Blade passing frequency, [Hz]
    flow_M = fan_mdot / (rho_0 * aircraft.engine.fan_A * c_0)  # Fan face flow Mach number (assumes ambient and fan face static densities are equal): !!!!!!!
    M_tip = (M_tip_tan ** 2 + flow_M ** 2) ** 0.5  # Relative (i.e., helical) tip Mach number: ! Doesn't need correction

    # Temperature-flow power base term:
    rho_sl = 1.22514
    c_sl = 340.29395
    if settings['fan_BB_method'] == 'kresja':
        tsqem = 10 * jjnp.log10((fan_DTt * 1.8) ** 4 * 2.20462 * fan_mdot / (1 - M_0 * jjnp.cos(theta * jjnp.pi / 180)) ** 4 / settings['r_0'] ** 2 / (rho_sl ** 2 * c_sl ** 4))
    else:  # All other methods:
        tsqem = 10 * jjnp.log10((fan_DTt * 1.8) ** 2 * 2.20462 * fan_mdot / (1 - M_0 * jjnp.cos(theta * jjnp.pi / 180)) ** 4 / settings['r_0'] ** 2 / (rho_sl ** 2 * c_sl ** 4))

    # Calculate individual noise components
    dcp = jjnp.zeros(settings['n_frequency_bands'])
    if comp == 'fan_inlet':
        bblv_I = inlet_broadband(settings, theta, M_tip, tsqem, aircraft.engine.fan_M_d, aircraft.engine.fan_fan_RSS)
        tonlv_I = inlet_tones(settings, theta, M_tip, tsqem, aircraft.engine.fan_M_d, aircraft.engine.fan_fan_RSS)
        bblv_D = 0.
        tonlv_X = 0.
    elif comp == 'fan_discharge':
        bblv_I = 0.
        tonlv_I = 0.
        bblv_D = discharge_broadband(settings, theta, M_tip, tsqem, aircraft.engine.fan_M_d, aircraft.engine.fan_fan_RSS)
        tonlv_X = discharge_tones(settings, theta, M_tip, tsqem, aircraft.engine.fan_M_d, aircraft.engine.fan_fan_RSS)
    else:
        raise ValueError('Invalid component specified. Specify "fan_inlet" or "fan discharge".')

    # Compute combination tones
    if settings['fan_combination_tones']:
        dcp = combination_tones(settings, f, theta, M_tip, bpf, tsqem)

    # Calculate if cut-off happens (1) or not (0)
    i_cut = calculate_cutoff(M_tip_tan, aircraft.engine.fan_B, aircraft.engine.fan_V)

    # Assign tones_to bands
    dp, dpx = calculate_harmonics(settings, f, theta, tonlv_I, tonlv_X, i_cut, M_tip, bpf,comp)

    # Final calculations;  cycle through frequencies and assign values:
    if settings['fan_BB_method'] == 'allied_signal':
        # Eqn 2 or Figure 3A:
        # if f[j] / bpf < 2:
        # FLOGinlet,exit = -10 * jjnp.log10(jjnp.exp(-0.35 * (jjnp.log(f[j] / bpf / 2.0) / jjnp.log(2.2)) ** 2))
        FLOGinlet = 2.445096095 * (jjnp.log(f / bpf / 2)) ** 2
        # elif f[j] / bpf > 2:
        # FLOGinlet,exit = -10 * jjnp.log10(jjnp.exp(-2.0 * (jjnp.log(f[j] / bpf / 2.0) / jjnp.log(2.2)) ** 2))
        FLOGinlet[f/bpf > 2] = (13.97197769 * (jjnp.log(f / bpf / 2)) ** 2)[f/bpf > 2]
        FLOGexit = FLOGinlet

    elif settings['fan_BB_method'] == 'kresja':
        # Eqn 2 or Figure 3A:
        # FLOGinlet = -10 * jjnp.log10(jjnp.exp(-0.5 * (jjnp.log(f[j] / bpf / 4) / jjnp.log(2.2)) ** 2))
        # Which may be simplified as:
        FLOGinlet = 3.4929944 * (jjnp.log(f / bpf / 4)) ** 2
        # FLOGexit = -10 * jjnp.log10(jjnp.exp(-0.5 * (jjnp.log(f[j] / bpf / 2.5) / jjnp.log(2.2)) ** 2))
        # Which may be simplified as:
        FLOGexit = 3.4929944 * (jjnp.log(f / bpf / 2.5)) ** 2

    else:
        # For the original or the GE large fan methods:
        # Eqn 2 or Figure 3A:
        # FLOGinlet,exit = -10 * jjnp.log10(jjnp.exp(-0.5 * (jjnp.log(f[j] / bpf / 2.5) / jjnp.log(2.2)) ** 2))
        # Which may be simplified as:
        FLOGinlet = 3.4929944 * (jjnp.log(f / bpf / 2.5)) ** 2
        FLOGexit = FLOGinlet

    # Add frequency distribution to the fan broadband noise and add the tones
    if comp == 'fan_inlet':
        msap = 10 ** (0.1 * (bblv_I - FLOGinlet))
        msap = msap + dp

    elif comp == 'fan_discharge':
        msap = 10 ** (0.1 * (bblv_D - FLOGexit))
        msap = msap + dpx

    # Add inlet combination tones if needed:
    if M_tip > 1. and settings['fan_combination_tones'] == True:
        msap = msap + dcp

    # Multiply for number of engines
    msap = msap * aircraft.n_eng

    # Fan liner suppression
    if settings['fan_liner_suppression']:
        if comp == 'fan_inlet':
            supp = tables.source.fan.supp_fi_f(theta, f).reshape(settings['n_frequency_bands'],)

        elif comp == 'fan_discharge':
            supp = tables.source.fan.supp_fd_f(theta, f).reshape(settings['n_frequency_bands'], )

        msap = supp * msap

    return msap_fan