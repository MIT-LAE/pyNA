.. _exampleNoiseEPNLTable:

Computing a noise EPNL table
============================

This example shows how to compute a noise EPNL table for all noise sources for an existing trajectory.

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

	# Run pyna
	py = pyna(settings=pyna_settings)
	table = py.compute_noise_epnl_table()

	# Print table
	print(table)

