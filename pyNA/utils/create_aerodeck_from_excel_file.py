import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

def create_aerodeck_from_excel_file(file_name_cl: str, file_name_cd: str, save_file: bool):

	# Read the date
	data_cl = pd.read_excel('../pyNA/cases/stca/aircraft/NASA all flap data.xlsx', sheet_name='CL')
	data_cd = pd.read_excel('../pyNA/cases/stca/aircraft/NASA all flap data.xlsx', sheet_name='CD')

	# Extract independent variables
	alpha = data_cl.values[2:,0]
	theta_slats = np.unique(data_cl.values[0, 1:])
	theta_flaps = np.unique(data_cl.values[1, 1:])

	# Extract dependent variables
	CL = np.zeros((np.size(alpha), np.size(theta_flaps), np.size(theta_slats)))
	CD = np.zeros((np.size(alpha), np.size(theta_flaps), np.size(theta_slats)))

	for i in np.arange(np.size(alpha)):
	    for j in np.arange(np.size(theta_slats)):
	        CL[i, :, j] = np.flip(data_cl.values[2+i, 1+j*np.size(theta_flaps):1+(j+1)*np.size(theta_flaps)])
	        CD[i, :, j] = np.flip(data_cd.values[2+i, 1+j*np.size(theta_flaps):1+(j+1)*np.size(theta_flaps)])
	    
    # Save numpy output files
	if save_file: 
	    np.save('../pyNA/cases/stca/aircraft/cl_stca.npy', CL)
	    np.save('../pyNA/cases/stca/aircraft/cd_stca.npy', CD)

	return None


if __name__ == "__main__":
	pass