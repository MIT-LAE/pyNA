levels_int
==========

class description
-----------------

.. automodule:: levels_int
   :members:
   :undoc-members:
   :show-inheritance:

.. warning::
	
	The Julia version of the *Levels* component, ``levels_int.jl``, is not documented yet. However, it is very similar to the Python version, ``levels.py``. More information and examples on how OpenMDAO.jl components work can be found `here <https://github.com/byuflowlab/OpenMDAO.jl/tree/master/examples>`_.

theory
------

The effective perceived noise level (EPNL) is calculated using the method described in ICAO Annex 16, Volume I \cite{ICAO2017}. Firstly, the tone-corrected perceived noise level signal, :math:`PNLT`, and the tone corrections matrix, :math:`C`, at the observer are interpolated at time steps :math:`\Delta t = 0.5s`. The effective perceived noise level, :math:`EPNL`, is given by: 

.. math::

    EPNL = \max (PNLT) + D

The duration correction, :math:`D`, is given by:

.. math::

    D = 10 \log_{10} \frac{1}{t_0} \sum_{i\in \mathcal{I}} 10^{\frac{PNLT}{10}} - \max (PNLT)


where :math:`t_0 = 10s` and $\mathcal{I}$ contains all points in the :math:`PNLT` time series that satisfy :math:`PNLT > PNLTM - 10`. Finally, $PNLTM$ includes a correction term in the case of one-third octave band-sharing:

.. math::

    PNLTM = PNLT[i_{\max}] + \Delta_c

where :math:`i_{\max}` is the index of the maximum value of the :math:`PNLT` time series. The correction term, :math:`\Delta_c`, is given by: 

.. math::

    \Delta_c = \begin{cases}
    C_{\textrm{avg}} = \frac{1}{5}\sum_{i=-2}^{2}C[i_{\max} + i] \quad & \textrm{if} \quad C_{\textrm{avg}} > C(i_{\max})\\
    0 \quad & \textrm{otherwise}\\
    \end{cases}

