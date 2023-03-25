import numpy as np
from pyNA.src.noise_model.python.calculate_geometry import compute_geometry
from pyNA.src.noise_model.python.source.fan_heidman import fan_heidman
from pyNA.src.noise_model.python.source.core_emmerling import core_emmerling
from pyNA.src.noise_model.python.source.jet_mixing_sae876 import jet_mixing_sae876
from pyNA.src.noise_model.python.source.jet_shock_sae876 import jet_shock_sae876
from pyNA.src.noise_model.python.source.airframe_fink import airframe_fink
from pyNA.src.noise_model.python.propagation.calculate_propagation_effects import calculate_propagation_effects
from pyNA.src.noise_model.python.level.calculate_aspl import calculate_aspl
from pyNA.src.noise_model.python.level.calculate_oaspl import calculate_oaspl
from pyNA.src.noise_model.python.level.calculate_spl import calculate_spl
from pyNA.src.noise_model.python.level.calculate_pnlt import calculate_pnlt
from pyNA.src.noise_model.python.level.calculate_level_int import calculate_level_int
import pdb


def calculate_noise(x, y, z, alpha, gamma, t_s, tau, M_0, c_0, T_0, rho_0, P_0, mu_0, I_0, 
                    fan_DTt, fan_mdot, fan_N,
                    core_mdot, core_Tt_i, core_Tt_j, core_Pt_i, turb_DTt_des, turb_rho_e, turb_c_e, turb_rho_i, turb_c_i,
                    jet_V, jet_rho, jet_A, jet_Tt, jet_M,
                    theta_flaps, I_lg,
                    shielding,
                    x_microphone,
                    f, f_sb, 
                    settings, aircraft, tables, 
                    optimization):

    """
    
    Parameters
    ----------

    Outputs
    -------
    level : float
        _

    """

    # Compute geometry between trajectory and observer
    r, theta, phi, beta, t_o, c_bar = compute_geometry(x, y, z, alpha, gamma, t_s, c_0, T_0, x_microphone)

    # Compute source strength
    msap_source = np.zeros(settings['n_frequency_bands'])

    if settings['fan_inlet_source']:
        msap_fan_inlet = fan_heidman(fan_DTt, fan_mdot, fan_N, M_0, c_0, rho_0, theta, f, settings, aircraft, tables, comp='fan_inlet')    
        msap_fan_inlet = msap_fan_inlet / (10 ** (shielding / 10.))
        msap_source += msap_fan_inlet

    if settings['fan_discharge_source']:
        msap_fan_discharge = fan_heidman(fan_DTt, fan_mdot, fan_N, M_0, c_0, rho_0, theta, f, settings, aircraft, tables, comp='fan_discharge')
        msap_source += msap_fan_discharge

    if settings['core_source']:
        msap_core = core_emmerling(core_mdot, core_Tt_i, core_Tt_j, core_Pt_i, turb_DTt_des, turb_rho_e, turb_c_e, turb_rho_i, turb_c_i, M_0, rho_0, c_0, T_0, P_0, theta, f, settings, aircraft, tables)
        if settings['case_name'] == 'nasa_stca_standard' and settings['core_jet_suppression']:
            if tau > 0.8:
                msap_core = 10.**(-2.3 / 10.) * msap_core
        msap_source += msap_core

    if settings['jet_mixing_source']:
        msap_jet_mixing = jet_mixing_sae876(jet_V, jet_Tt, jet_A, jet_rho, M_0, c_0, T_0, rho_0, theta, f, settings, aircraft, tables)
        if settings['case_name'] == 'nasa_stca_standard' and settings['core_jet_suppression']:
            if tau > 0.8:
                msap_jet_mixing = 10.**(-2.3 / 10.) * msap_jet_mixing
        msap_source += msap_jet_mixing

    if settings['jet_shock_source']:
        msap_jet_shock = jet_shock_sae876(jet_V, jet_Tt, jet_A, jet_M, M_0, c_0, T_0, theta, f, settings, aircraft, tables)
        if settings['case_name'] == 'nasa_stca_standard' and settings['core_jet_suppression']:
            if tau > 0.8:
                msap_jet_shock = 10.**(-2.3 / 10.) * msap_jet_shock
        msap_source += msap_jet_shock
    
    if settings['airframe_source']:
        msap_airframe = airframe_fink(theta_flaps, I_lg, M_0, c_0, rho_0, mu_0, theta, phi, f, settings, aircraft, tables)
        msap_source += msap_airframe

    msap_source = msap_source.clip(min=1e-99)

    # Compute propagated effects
    msap_prop = calculate_propagation_effects(msap_source, z, r, c_bar, rho_0, I_0, beta, x_microphone, f_sb, settings, tables)

    # Compute spl
    spl = calculate_spl(msap_prop, rho_0, c_0)

    # Compute oaspl
    oaspl = calculate_oaspl(spl)

    # Compute pnlt
    pnlt, C = calculate_pnlt(spl, settings)
        
    # Compute aspl
    aspl = calculate_aspl(spl, f, tables)
    
    return t_o, spl, aspl, oaspl, pnlt