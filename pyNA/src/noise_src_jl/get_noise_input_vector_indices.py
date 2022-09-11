from typing import Dict


def get_input_vector_indices(language, settings: dict, n_t: int) -> Dict:
        
        """
        Get indices for input vector of noise model.

        :param settings: noise settings
        :type settings: dict
        :param n_t: number of time steps in trajectory
        :type n_t: int

        :return: idx

        """

        # Initialize indices dictionary
        idx = dict()

        if language == 'julia':
            idx["x"] = [0 * n_t + 1, 1 * n_t]
            idx["y"] = [1 * n_t + 1, 2 * n_t]
            idx["z"] = [2 * n_t + 1, 3 * n_t]
            idx["alpha"] = [3 * n_t + 1, 4 * n_t]
            idx["gamma"] = [4 * n_t + 1, 5 * n_t]
            idx["t_s"] = [5 * n_t + 1, 6 * n_t]
            idx["rho_0"] = [6 * n_t + 1, 7 * n_t]
            idx["mu_0"] = [7 * n_t + 1, 8 * n_t]
            idx["c_0"] = [8 * n_t + 1, 9 * n_t]
            idx["T_0"] = [9 * n_t + 1, 10 * n_t]
            idx["p_0"] = [10 * n_t + 1, 11 * n_t]
            idx["M_0"] = [11 * n_t + 1, 12 * n_t]
            idx["I_0"] = [12 * n_t + 1, 13 * n_t]
            idx["TS"] = [13 * n_t + 1, 14 * n_t]
            idx["theta_flaps"] = [14 * n_t + 1, 15 * n_t]
            n = 15
            if settings['jet_mixing_source'] == True and settings['jet_shock_source'] == False:
                idx["V_j"]   = [n * n_t + 1, (n + 1) * n_t]
                idx["rho_j"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                idx["A_j"]   = [(n + 2) * n_t + 1, (n + 3) * n_t]
                idx["Tt_j"]  = [(n + 3) * n_t + 1, (n + 4) * n_t]
                n = n + 4
            elif settings['jet_shock_source'] == True and settings['jet_mixing_source'] == False:
                idx["V_j"]  = [n * n_t + 1, (n + 1) * n_t]
                idx["M_j"]       = [(n + 1) * n_t + 1, (n + 2) * n_t]
                idx["A_j"]  = [(n + 2) * n_t + 1, (n + 3) * n_t]
                idx["Tt_j"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                n = n + 4
            elif settings['jet_shock_source'] == True and settings['jet_mixing_source'] == True:
                idx["V_j"] = [n * n_t + 1, (n + 1) * n_t]
                idx["rho_j"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                idx["A_j"] = [(n + 2) * n_t + 1, (n + 3) * n_t]
                idx["Tt_j"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                idx["M_j"] = [(n + 4) * n_t + 1, (n + 5) * n_t]
                n = n + 5
            if settings['core_source']:
                if settings['core_turbine_attenuation_method'] == "ge":
                    idx["mdoti_c"] = [n * n_t + 1, (n + 1) * n_t]
                    idx["Tti_c"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                    idx["Ttj_c"] = [(n + 2) * n_t + 1, (n + 3) * n_t]
                    idx["Pti_c"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                    idx["DTt_des_c"] = [(n + 4) * n_t + 1, (n + 5) * n_t]
                    n = n + 5
                elif settings['core_turbine_attenuation_method'] == "pw":
                    idx["mdoti_c"] = [n * n_t + 1, (n + 1) * n_t]
                    idx["Tti_c"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                    idx["Ttj_c"] = [(n + 2) * n_t + 1, (n + 3) * n_t]
                    idx["Pti_c"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                    idx["rho_te_c"] = [(n + 4) * n_t + 1, (n + 5) * n_t]
                    idx["c_te_c"] = [(n + 5) * n_t + 1, (n + 6) * n_t]
                    idx["rho_ti_c"] = [(n + 6) * n_t + 1, (n + 7) * n_t]
                    idx["c_ti_c"] = [(n + 7) * n_t + 1, (n + 8) * n_t]
                    n = n + 8
            if settings['airframe_source']:
                idx["I_landing_gear"] = [n * n_t + 1, (n + 1) * n_t]
                n = n + 1
            if settings['fan_inlet_source'] == True or settings['fan_discharge_source'] == True:
                idx["DTt_f"] = [n * n_t + 1, (n + 1) * n_t]
                idx["mdot_f"] = [(n + 1) * n_t + 1, (n + 2) * n_t]
                idx["N_f"] = [(n + 2) * n_t + 1, (n + 3) * n_t]
                idx["A_f"] = [(n + 3) * n_t + 1, (n + 4) * n_t]
                idx["d_f"] = [(n + 4) * n_t + 1, (n + 5) * n_t]

        elif language == 'python':
            idx["x"] = [0 * n_t, 1 * n_t]
            idx["y"] = [1 * n_t, 2 * n_t]
            idx["z"] = [2 * n_t, 3 * n_t]
            idx["alpha"] = [3 * n_t, 4 * n_t]
            idx["gamma"] = [4 * n_t, 5 * n_t]
            idx["t_s"] = [5 * n_t, 6 * n_t]
            idx["rho_0"] = [6 * n_t, 7 * n_t]
            idx["mu_0"] = [7 * n_t, 8 * n_t]
            idx["c_0"] = [8 * n_t, 9 * n_t]
            idx["T_0"] = [9 * n_t, 10 * n_t]
            idx["p_0"] = [10 * n_t, 11 * n_t]
            idx["M_0"] = [11 * n_t, 12 * n_t]
            idx["I_0"] = [12 * n_t, 13 * n_t]
            idx["TS"] = [13 * n_t, 14 * n_t]
            idx["theta_flaps"] = [14 * n_t, 15 * n_t]
            n = 15
            if settings['jet_mixing_source'] == True and settings['jet_shock_source'] == False:
                idx["V_j"]   = [n * n_t, (n + 1) * n_t]
                idx["rho_j"] = [(n + 1) * n_t, (n + 2) * n_t]
                idx["A_j"]   = [(n + 2) * n_t, (n + 3) * n_t]
                idx["Tt_j"]  = [(n + 3) * n_t, (n + 4) * n_t]
                n = n + 4
            elif settings['jet_shock_source'] == True and settings['jet_mixing_source'] == False:
                idx["V_j"]  = [n * n_t, (n + 1) * n_t]
                idx["M_j"]  = [(n + 1) * n_t, (n + 2) * n_t]
                idx["A_j"]  = [(n + 2) * n_t, (n + 3) * n_t]
                idx["Tt_j"] = [(n + 3) * n_t, (n + 4) * n_t]
                n = n + 4
            elif settings['jet_shock_source'] == True and settings['jet_mixing_source'] == True:
                idx["V_j"] = [n * n_t, (n + 1) * n_t]
                idx["rho_j"] = [(n + 1) * n_t, (n + 2) * n_t]
                idx["A_j"] = [(n + 2) * n_t, (n + 3) * n_t]
                idx["Tt_j"] = [(n + 3) * n_t, (n + 4) * n_t]
                idx["M_j"] = [(n + 4) * n_t, (n + 5) * n_t]
                n = n + 5
            if settings['core_source']:
                if settings['core_turbine_attenuation_method'] == "ge":
                    idx["mdoti_c"] = [n * n_t, (n + 1) * n_t]
                    idx["Tti_c"] = [(n + 1) * n_t, (n + 2) * n_t]
                    idx["Ttj_c"] = [(n + 2) * n_t, (n + 3) * n_t]
                    idx["Pti_c"] = [(n + 3) * n_t, (n + 4) * n_t]
                    idx["DTt_des_c"] = [(n + 4) * n_t, (n + 5) * n_t]
                    n = n + 5
                elif settings['core_turbine_attenuation_method'] == "pw":
                    idx["mdoti_c"] = [n * n_t, (n + 1) * n_t]
                    idx["Tti_c"] = [(n + 1) * n_t, (n + 2) * n_t]
                    idx["Ttj_c"] = [(n + 2) * n_t, (n + 3) * n_t]
                    idx["Pti_c"] = [(n + 3) * n_t, (n + 4) * n_t]
                    idx["rho_te_c"] = [(n + 4) * n_t, (n + 5) * n_t]
                    idx["c_te_c"] = [(n + 5) * n_t, (n + 6) * n_t]
                    idx["rho_ti_c"] = [(n + 6) * n_t, (n + 7) * n_t]
                    idx["c_ti_c"] = [(n + 7) * n_t, (n + 8) * n_t]
                    n = n + 8
            if settings['airframe_source']:
                idx["I_landing_gear"] = [n * n_t, n * n_t]
                n = n + 1
            if settings['fan_inlet_source'] == True or settings['fan_discharge_source'] == True:
                idx["DTt_f"] = [n * n_t, (n + 1) * n_t]
                idx["mdot_f"] = [(n + 1) * n_t, (n + 2) * n_t]
                idx["N_f"] = [(n + 2) * n_t, (n + 3) * n_t]
                idx["A_f"] = [(n + 3) * n_t, (n + 4) * n_t]
                idx["d_f"] = [(n + 4) * n_t, (n + 5) * n_t]

        return idx