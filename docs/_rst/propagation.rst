propagation
===========

class description
-----------------

.. automodule:: propagation
   :members:
   :undoc-members:
   :show-inheritance:

.. warning::
	
	The Julia version of the *Propagation* component, ``propagation.jl``, is not documented yet. However, it is very similar to the Python version, ``propagation.py``. More information and examples on how OpenMDAO.jl components work can be found `here <https://github.com/byuflowlab/OpenMDAO.jl/tree/master/examples>`_.


theory
------

The propagation of noise through the atmosphere is composed of 4 effects. Firstly, the noise power away from the source reduces with the distance squared, i.e. the :math:`R^2`-law, as the noise power is distributed over outward moving spherical surfaces. Secondly, the temperature gradient in the troposphere causes a difference between the characteristic impedance at the source relative to that at the observer. Thirdly, while the sound waves are propagating through the atmosphere, the noise power decreases exponentially, because of atmospheric absorption. Next, surface absorption and reflections need to be taken into account when considering a microphone close to the ground. Finally, the lateral noise attenuation at low elevation angles as well as engine installation effects have to be taken into account. Note that the propagation effects in this section are applied to the mean-square acoustic pressure itself (:math:`<p^2>` instead of :math:`<p^2>^*`).

Spherical spreading and characteristic impedance effect
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the mean-square pressure level at the source, :math:`<p^2_{\textrm{source}}>^*`, the directly-propagated mean-square pressure level at the observer, :math:`<p^2_{\textrm{direct-prop}}>^*`, is calculated using:

.. math::

    <p^2_{\textrm{direct-prop}}> = \underbrace{\left[\frac{r_{\textrm{s}}^2}{r^2}\right]}_{\substack{\text{$R^2$} \\ \text{law}}} \cdot \underbrace{\left[\frac{(I_0)_{\textrm{obs}}}{(I_0)_{\textrm{source}}} \right]}_{\substack{\text{characteristic acoustic} \\ \text{impedance effect}}} <p^2_{\textrm{source}}>

Atmospheric absorption effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The atmospheric absorption effects are applied to the mean-square acoustic pressure of each sub-band :math:`j` using:

.. math::

    <p^2_{\textrm{absorb}}*_j \ = \ \underbrace{\exp{\left[-2\alpha(r-r_{\textrm{source}})\right]}}_{\textrm{atmospheric absorption}} <p^2_{\textrm{direct-prop}}>_j

A look-up table of the atmospheric absorption coefficient, :math:`\alpha`, in the US Standard Atmosphere is implemented as a function of altitude, :math:`z`, and sub-band frequency, :math:`f_{sb}`.


Combination of sub-bands
^^^^^^^^^^^^^^^^^^^^^^^^

After applying the atmospheric absorption and ground effects, the :math:`N_b` sub-bands are combined again to the original frequency band :math:`i` and the resulting mean-square acoustic pressure is converted back to decibels using: 

.. math::

    <p^2_{\textrm{propagated}}>^*_i = 10\log_{10}\left[\sum_{h = 1}^{N_b} <p^2_{\textrm{propagated}}>_h \right]

Shielding module
^^^^^^^^^^^^^^^^

The Maekawa method is used to assess the wing shielding effect on the engine noise. This is not implemented yet in pyNA.

