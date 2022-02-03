.. _installation:

installation
============

Python version of pyNA
----------------------

Get pyNA from a cloned repository from `Github <https://github.mit.edu/lvoet/pyNA>`_:

	git clone git@github.mit.edu:lvoet/pyNA.git 

Use pip to install pyNA:

	pip install -e .

The python version is default in pyNA. To enable this mode, set a python environment variable to julia:

.. code-block::
 	
 	import os
 	os.environ['pyna_language'] = 'python'

Julia version of pyNA
---------------------

To enable fast computation of sensitivities of acoustic objective functions in pyNA, an installation of Julia is required since the modules *geometry, source, propagation and levels* are using Julia's ForwardDiff. Install `Julia <https://julialang.org>`_. To enable this mode, set a python environment variable to julia:

.. code-block::
 	
 	import os
 	os.environ['pyna_language'] = 'julia'
