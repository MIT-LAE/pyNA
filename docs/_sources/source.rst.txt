source
======

class description
-----------------

.. automodule:: source
   :members:
   :undoc-members:
   :show-inheritance:

.. warning::
	
	The Julia version of the *Source* component, ``source.jl``, is not documented yet. However, it is very similar to the Python version, ``source.py``. More information and examples on how OpenMDAO.jl components work can be found `here <https://github.com/byuflowlab/OpenMDAO.jl/tree/master/examples>`_.

.. note::

	The mean-square acoustic pressure at the source, :math:`<p^2_{source}>`, is normalized by a reference pressure, :math:`p_{\textrm{ref}} = 2\cdot 10^{-5}` Pa and is computed in decibels: 

	.. math::

	    <p^2_{\textrm{source}}>^* = 10\log_{10}\frac{<p^2_{\textrm{source}}>}{p_{ref}^2}
