import pandas as pd
import numpy as np

# Define read function
def read_NPSS_file(file_name: str, save_file: bool, rows_to_skip = 1):
    
    df = pd.read_csv(filename, delimiter=r'\s+', skiprows=1)
    
    return df

def create_engine_deck_from_NPSS(file_name: str):

	# Read NPSS file
	filename='../pyNA/cases/stca/engine/STCA_Deriv_EngineDeck.out'
	data = read_NPSS(filename)

	# Create engine deck
	engine = pd.DataFrame()

	engine['z [m]'] = 0.3048*data['Alt[ft]']
	engine['M_0 [-]'] = data['MN']
	engine['T/TMAX [-]'] = data['ThrustSetting']/100.
	engine['Nc_L [%]'] = np.zeros(np.shape(data['fan_A[m2]']))
	engine['Fn [N]'] = data['NetThrust[N]']
	engine['Wf [kg/s]'] = data['Wf[kg/s]']
	engine['jet_V [m/s]'] = data['jet_V[m/s]'].values
	engine['jet_Tt [K]'] = data['jet_Tt[K]']
	engine['jet_rho [kg/m3]'] = data['jet_rho[kg/m3]']
	engine['jet_A [m2]'] = data['jet_A[m2]']
	engine['jet_M [-]'] = data['jet_M[-]']
	engine['core_mdot_in [kg/s]'] = data['core_mdot_in[kg/s]']
	engine['core_Pt_in [Pa]'] = data['core_Pt_in[Pa]']
	engine['core_Tt_in [K]'] = data['core_Tt_in[K]']
	engine['core_Tt_out [K]'] = data['core_Tt_out[K]']
	engine['core_DT_t [K]'] = data['core_DT_t[K]']
	engine['core_HPT_rho_in [kg/m3]'] = data['core_HPT_rho_in[kg/m3]']
	engine['core_HPT_c_in [m/s]'] = data['core_HPT_c_in[m/s]']
	engine['core_LPT_rho_out [kg/m3]'] = data['core_LPT_rho_out[kg/m3]']
	engine['core_LPT_c_out [m/s]'] = data['core_LPT_c_out[m/s]']
	engine['fan_mdot_in [kg/s]'] = data['fan_mdot_in[kg/s]']
	engine['fan_N [rpm]'] = data['fan_N[rpm]']
	engine['fan_DTt [K]'] = data['fan_DT[K]']
	engine['fan_d [m]'] = data['fan_d[m]']
	engine['fan_A [m2]'] = data['fan_A[m2]']
	engine['fan_M_d [-]'] = np.zeros(np.shape(data['fan_A[m2]']))
	engine['fan_B [-]'] = np.zeros(np.shape(data['fan_A[m2]']))
	engine['fan_V [-]'] = np.zeros(np.shape(data['fan_A[m2]']))
	engine['fan RSS [-]'] = np.zeros(np.shape(data['fan_A[m2]']))

	if save_file:
		engine.to_csv('../pyNA/cases/stca/engine/engine_deck_stca.csv', index=False)

	return 


if __name__ == "__main__":
	pass
