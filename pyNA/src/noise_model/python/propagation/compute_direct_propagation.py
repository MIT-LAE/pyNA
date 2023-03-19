import pdb


def compute_direct_propagation(msap_source, r, I_0, settings):

    """
    
    :param msap_source:
    :type msap_source:
    :param r:
    :type r:
    :param I_0:
    :type I_0:
    :param settings:
    :type settings:
    """

    I_0_obs = 409.74  # Sea level characteristic impedance [kg/m**2/s]
    msap_prop = msap_source * (settings['r_0']/r)**2 * (I_0_obs / I_0)

    return msap_prop