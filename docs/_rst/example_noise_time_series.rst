.. _exampleNoiseTimeSeries:

Computing a noise time series
=============================

This example shows how to compute a noise time series from an existing trajectory.

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
	pyna_settings = pyna.load_settings(case_name ='nasa_stca_standard')
	pyna_settings.save_results = True

	# Run pyna
	py = pyna(settings=pyna_settings)
	py.compute_noise_time_series()

	# Plot results
	py.plot_noise_time_series(metric='pnlt')