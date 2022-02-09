.. _exampleNoiseContours:

Computing noise contours
========================

This example shows how to compute ground level noise contours around an existing trajectory.

.. code-block:: python

	import os
	import pdb
	import numpy as np
	import openmdao.api as om
	import pandas as pd
	import matplotlib.pyplot as plt

	os.environ["pyna_language"] = 'python'
	from pyNA.pyna import pyna

	# Load default pyna settings
	pyna_settings = pyna.load_settings(case_name = 'nasa_stca_standard')
	pyna_settings.save_results = True

	# Compute noise contours
	x_lst = np.linspace(0, 15000, 16)
	y_lst = np.linspace(0, 4000, 5)
	py = pyna(settings=pyna_settings)
	py.compute_noise_contours(x_lst=x_lst, y_lst=y_lst)

	# Plot
	py.plot_noise_contours()