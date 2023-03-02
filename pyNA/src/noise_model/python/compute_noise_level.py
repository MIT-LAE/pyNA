import jax.numpy as jnp
from pyNA.src.noise_model.python.compute_geometry import compute_geometry
from pyNA.src.noise_model.python.source.fan_heidman import fan_heidman
from pyNA.src.noise_model.python.source.core_emmerling import core_emmerling
from pyNA.src.noise_model.python.source.jet_mixing_sae876 import jet_mixing_sae876
from pyNA.src.noise_model.python.source.jet_shock_sae876 import jet_shock_sae876
from pyNA.src.noise_model.python.source.airframe_fink import airframe_fink

def compute_noise_level(x, y, z, alpha, gamma, t_s, tau, M_0, c_0, T_0, rho_0, p_0, mu_0, I_0, 
                        fan_DTt, fan_mdot, fan_N,
                        core_mdot, core_Tt_i, core_Tt_j, core_Pt_i, turb_DTt_des, turb_rho_e, turb_c_e, turb_rho_i, turb_c_i,
                        jet_V, jet_rho, jet_A, jet_Tt, jet_M,
                        x_mic,
                        settings):

    # Compute geometry between trajectory and observer
    r, theta, phi, beta, t_o, c_bar = compute_geometry(x, y, z, alpha, gamma, t_s, c_0, T_0, x_mic)

    # Compute source strength
    msap_source = jnp.zeros(settings['n_frequency_bands'])

    if settings['fan_inlet_source']:
        msap_fan_inlet = fan_heidman()
        if settings['case_name'] == 'nasa_stca_standard' and settings['shielding']:
            shielding = 1
            msap_fan_inlet = msap_fan_inlet / (10 ** (shielding[i, :] / 10.))
        msap_source += msap_fan_inlet

    if settings['fan_discharge_source']:
        msap_fan_discharge = fan_heidman()
        msap_source += msap_fan_discharge
    if settings['core_source']:
        msap_core = core_emmerling()
        msap_source += msap_core

    if settings['jet_mixing_source']:
        msap_jet_mixing = jet_mixing_sae876()
        msap_source += msap_jet_mixing

    if settings['jet_shock_source']:
        msap_jet_shock = jet_shock_sae876()
        msap_source += msap_jet_shock
    
    if settings['airframe_source']:
        msap_airframe = airframe_fink()
        msap_source += msap_airframe

    # Compute propagated effects
    msap_prop = 1

    # Compute noise levels
    level = 1

    return level