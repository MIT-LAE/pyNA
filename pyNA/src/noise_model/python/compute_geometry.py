import numpy as np

def compute_geometry(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic):

    # Geometry calculations
    # Compute body angles (psi_B, theta_B, phi_B): angle of body w.r.t. horizontal
    theta_B = alpha + gamma
    phi_B = np.zeros(alpha.shape)
    psi_B = np.zeros(alpha.shape)

    # Compute the relative microphone-source position vector i.e. difference between microphone and ac coordinate
    # Note: add 4 meters to the alitude of the aircraft (for engine height)
    r_0 =  x_mic[0] - x
    r_1 =  x_mic[1] - y
    r_2 = -x_mic[2] + (z + 4)

    # Compute the distance of the microphone-source vector
    r = np.sqrt(r_0 ** 2 + r_1 ** 2 + r_2 ** 2)
    
    # Normalize the distance vector
    # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 17
    n_vcr_a_0 = r_0 / r
    n_vcr_a_1 = r_1 / r
    n_vcr_a_2 = r_2 / r

    # Define elevation angle
    # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 21
    beta = 180. / np.pi * np.arcsin(n_vcr_a_2)

    # Transformation direction cosines (Euler angles) to the source coordinate system (i.e. take position of the aircraft into account)
    # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 22-25
    cth  = np.cos(np.pi / 180. * theta_B)
    sth  = np.sin(np.pi / 180. * theta_B)
    cphi = np.cos(np.pi / 180. * phi_B)
    sphi = np.sin(np.pi / 180. * phi_B)
    cpsi = np.cos(np.pi / 180. * psi_B)
    spsi = np.sin(np.pi / 180. * psi_B)

    n_vcr_s_0 = cth * cpsi * n_vcr_a_0 + cth * spsi * n_vcr_a_1 - sth * n_vcr_a_2
    n_vcr_s_1 = (-spsi * cphi + sphi * sth * cpsi) * n_vcr_a_0 + ( cphi * cpsi + sphi * sth * spsi) * n_vcr_a_1 + sphi * cth * n_vcr_a_2
    n_vcr_s_2 = (spsi * sphi + cphi * sth * cpsi) * n_vcr_a_0 + ( -sphi * cpsi + cphi * sth * spsi) * n_vcr_a_1 + cphi * cth * n_vcr_a_2

    # Compute polar directivity angle
    # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 26
    theta = 180. / np.pi * np.arccos(n_vcr_s_0)
    
    # Compute azimuthal directivity angle
    # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 27
    phi = -180. / np.pi * np.arctan2(n_vcr_s_1, n_vcr_s_2)
    
    # Compute average speed of sound between source and microphone
    n_intermediate = 11
    dz = z / n_intermediate
    c_bar = c_0
    for k in np.arange(1, n_intermediate):
        T_im = T_0 - k * dz * (-0.0065)
        c_im = np.sqrt(1.4 * 287. * T_im)
        c_bar = (k) / (k + 1) * c_bar + c_im / (k + 1)

    # Compute observed time
    # Source: Zorumski report 1982 part 1. Chapter 2.2 Equation 20
    t_o = t_s + r / c_bar

    return r, theta, phi, beta, t_o, c_bar