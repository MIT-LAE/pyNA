fan
===

class description
-----------------

.. automodule:: fan
	:members:
	:undoc-members:
	:show-inheritance:
	:exclude-members:

theory
------

The Heidmann method with GEAE revision is used to assess the single-stage fan broadband noise. Single-stage fan rotor-stator interaction tones are assessed using the Heidmann method with the Allied Signal revision. The aforementioned methods distinguish between forward radiated noise (i.e. inlet noise) and rearward radiated noise (i.e. discharge noise), resulting in 4 noise components to be estimated: inlet broadband, inlet tones, discharge broadband and discharge tones. The broadband level, bblv, and tone level, tonlv, of these components is computed using: 

.. math::

	\begin{aligned}
	bblv_{comp} =& \ T + F_1 + F_2 + F_3 + C + F_{\textrm{freq}}\\ 
	tonlv_{comp} =& \ T + F_1 + F_2 + F_3 + C \\ 
	\end{aligned}


where the term, :math:`T`, accounting for the effects of fan temperature rise, fan mass flow as well as the Doppler effect (:math:`(1-M_0 cos\theta)`), is given by: 

.. math::

    T = 10\log_{10} \left[ \frac{1}{\rho_{sl}^2 c_{sl}^4} \frac{(1.8 \Delta T_{fan}^* T_0)^2}{(r_s^*)^2} \frac{(2.20462\ \dot{m}^* \rho_0 c_0 A_e)}{(1 - M_0 \cos \theta )^4 } \right]


The tip Mach number correction term, :math:`F_{1}`, the rotor-stator spacing correction term, :math:`F_{2}`, the polar directivity correction term, :math:`F_{3}:math:`, the inlet guide vane correction term, :math:`C`, and the broadband frequency dependent term, :math:`F_{\textrm{freq}}`, are defined for the different noise components in the following sections. The tip Mach number :math:`M_{tip} = \sqrt{ M_{flow}^2 + M_{tip,tan}^2}` where :math:`M_{flow} = \frac{\dot{m}^*}{A^*}` and :math:`M_{tip,tan} = \pi N^*`. Note that the coefficients in the equation above are used for changes of units of the temperature and mass flow terms (1.8 to transform from K to R and 2.20462 from kg/s to lbm/s).

Inlet broadband (IB) noise component
""""""""""""""""""""""""""""""""""""
The tip Mach number dependent term for the inlet broadband noise component is defined by:

.. math::
	
	F_{1,IB} = 
	\begin{cases}
	58.5 & \textrm{if $M_{tip,des} \leq 1$ and $M_{tip} \leq 0.9$} \\
	58.5 - 50 \log_{10} \left(\frac{M_{tip}}{0.9}\right) & \textrm{if $M_{tip,des} \leq 1$ and $M_{tip} > 0.9$}\\
	58.5 + 20 \log_{10} \left(\frac{M_{tip,des}}{0.9}\right) & \textrm{if $M_{tip,des} > 1$ and $M_{tip} \leq 0.9$}\\
	58.5 + 20 \log_{10} \left(\frac{M_{tip,des}}{0.9}\right) - 50\log_{10}\left(\frac{M_{tip}}{0.9}\right) & \textrm{if $M_{tip,des} > 1$ and $M_{tip} > 0.9$}\\
	\end{cases}


The rotor-stator spacing correction term for inlet broadband noise :math:`F_{3,IB} = 0`. The :math:`\theta`-correction term, :math:`F_{3,IB}`, is tabulated online. The inlet guide vane correction term C = 0dB. The spectral distribution function for fan inlet noise is given by:

.. math::

    F_{\textrm{freq, inlet}} = 10 \log_{10} \left[ \exp \left[ -0.5 \left(\frac{\log \frac{f}{2.5f_b}}{\log 2.2} \right)^2 \right] \right] = -3.49299 \left(\log \frac{f}{2.5 f_b} \right)^2

Discharge broadband (DB) noise component
""""""""""""""""""""""""""""""""""""""""

The tip Mach number dependent term for the inlet broadband noise component is defined by:

.. math::
	
	F_{1,DB} = 
	\begin{cases}
	63.0 & \textrm{if $M_{tip,des} \leq 1$ and $M_{tip} \leq 1$} \\
	63.0 - 30 \log_{10} \left(M_{tip}\right)  & \textrm{if $M_{tip,des} \leq 1$ and $M_{tip} > 1$}\\
	63.0 + 20 \log_{10} \left(M_{tip,des}\right)  & \textrm{if $M_{tip,des} > 1$ and $M_{tip} \leq 1$}\\
	63.0 + 20 \log_{10} \left(M_{tip,des}\right) - 30\log_{10}\left(M_{tip}\right) & \textrm{if $M_{tip,des} > 1$ and $M_{tip} > 1$}\\
	\end{cases}

The rotor-stator spacing correction term is given by :math:`F_{2,IB} = -5 \log_{10}\left( \frac{RSS}{300} \right)`. The :math:`\theta`-correction term (:math:`F_{3,DB})` is tabulated online. If the fan stage has inlet guide vanes, the correction term  C = 3dB. The spectral distribution function for fan discharge noise is given by the same equation as for inlet broadband noise.


Inlet tones (IT) noise component
""""""""""""""""""""""""""""""""

The tip Mach number dependent term for the inlet tones noise component is defined by:

.. math::

	F_{1,IT}  = 
	\begin{cases}
	54.5 & \textrm{if $M_{tip,des} \leq 1$ and $M_{tip} \leq 0.72$} \\
	\min\left[54.5 + 50 \log_{10} \left(\frac{M_{tip}}{0.72}\right) , 53.5 + 80 \log_{10} \left(\frac{1}{M_{tip}}\right) \right]  & \textrm{if $M_{tip,des} \leq 1$ and $M_{tip} > 0.72$}\\
	54.5 + 20 \log_{10} \left(M_{tip,des}\right)  & \textrm{if $M_{tip,des} > 1$ and $M_{tip} \leq 0.72$}\\
	\min\Bigl[54.5 + 20 \log_{10} \left(M_{tip}\right) + 50\log_{10}\left(\frac{M_{tip}}{0.72}\right) , & \\ 53.5 + 80 \log_{10} \left(\frac{M_{tip,des}}{M_{tip}}\right) \Bigr]  & \textrm{if $M_{tip,des} \leq 1$ and $M_{tip} > 0.72$}\\
	\end{cases}

The rotor-stator spacing correction term is given by :math:`F_{2,TI} = -10 \log_{10}\left( \frac{RSS}{300} \right)`. The :math:`\theta` correction term (:math:`F_{3,IT}`) is tabulated online. The inlet guide vane correction term C = 0dB.

Discharge tones (DT) noise component
""""""""""""""""""""""""""""""""""""

The tip Mach number dependent term for the inlet tones noise component is defined by:

.. math::

	F_{1,DT} = 
	\begin{cases}
	59.0 & \textrm{if $M_{tip,des} \leq 1$ and $M_{tip} \leq 1$} \\
	59.0 - 20 \log_{10} \left(M_{tip}\right)  & \textrm{if $M_{tip,des} \leq 1$ and $M_{tip} > 1$}\\
	59.0 + 20 \log_{10} \left(M_{tip,des}\right)  & \textrm{if $M_{tip,des} > 1$ and $M_{tip} \leq 1$}\\
	59.0 + 20 \log_{10} \left(M_{tip,des}\right) - 20\log_{10}\left(M_{tip}\right) & \textrm{if $M_{tip,des} > 1$ and $M_{tip} > 1$}\\
	\end{cases}

The rotor-stator spacing correction term is given by :math:`F_{2,DT} = -10 \log_{10}\left( \frac{RSS}{300} \right)`. The :math:`\theta` correction term (:math:`F_{3,DT}`) is tabulated online. If the fan stage has inlet guide vanes, the correction term  C = 6dB.


Harmonics
"""""""""

The fundamental fan tone occurs at the blade pass frequency (bfp), given by: 

.. math::

    bpf = \frac{N_f^* c_0 B_f}{d_f(1 - M_0 \cos\theta)}

where :math:`B_f` is the number of rotor blades, :math:`d_f` is the fan diameter and :math:`(1 - M_0 \cos\theta)` is the Doppler correction factor. The level of the fundamental tone harmonics are determined by the harmonic fall-off rates. These fall-off rates are driven by the fan cut-off phenomenon. The fundamental cutoff factor, :math:`\delta`, at which the magnitude of the fan tones is attenuated is determined by the ratio between the number of stator vanes, :math:`V_f`, and rotor blades, :math:`B_f`:

.. math::

    \delta =  \frac{M_{tip}}{|1 - \frac{V_f}{B_f}|}

The cut-off index, :math:`i_{cut}`, is computed by: 

.. math::

    \begin{cases}
    i_{cut} = 1 \quad & \textrm{if}\ \quad \delta < 1.05 \ \textrm{and}\ M_{tip, tan} < 1  \\ 
    i_{cut} = 0 \quad & \textrm{if}\ \quad \textrm{otherwise} \\ 
    \end{cases}

The fall-off rates for the k-th harmonic of the inlet and discharge tones are given by:

.. math::

    \begin{cases}
    L_{\textrm{fall}} = 8 & \textrm{if}\ i_{cut} = 1\ \textrm{and}\ k = 1 \\
    L_{\textrm{fall}} = 0 & \textrm{if}\ i_{cut} = 0\ \textrm{and}\ k = 1 \\
    L_{\textrm{fall}} = 9.2 & \textrm{if}\ k = 2 \\
    L_{\textrm{fall}} = 3k + 1.8 & \textrm{if}\ k \geq 3 \\
    \end{cases}

The fall-off rate for each harmonic is subtracted from the tone level of each harmonic :math:`k \in [1, N_{harmonics}]`. Subsequently, the tone-level of each of the harmonics have to be added to correct one-third octave frequency band. A one-third octave frequency band extends between :math:`[2^{-1/6}f_c, 2^{1/6}f_c]`Hz. Adding all harmonics to the correct frequency bands results in the tone matrices :math:`T_I` and :math:`T_{D}`.

Combining of the fan noise components
"""""""""""""""""""""""""""""""""""""

The fan inlet and discharge mean-square acoustic pressure is computed by combining the broadband spectrum and rotor-stator interaction tones:

.. math::

    \begin{aligned}
    <p^2>^*_{\textrm{inlet}} & = bblv_I + T_I \\
    <p^2>^*_{\textrm{discharge}} & = bblv_D + T_D \\
    \end{aligned}

Fan liner noise suppression
"""""""""""""""""""""""""""

The effect of fan liner treatment on the overall fan noise is assessed using the NASA CR-202309 method. The fan suppression tables for inlet and discharge noise are tabulated online as a function of frequency, :math:`f`, and polar directivity angle, :math:`\theta`. 

.. math::

	\begin{aligned}
	    <p^2>^*_{\textrm{inlet, supp.}} & = \quad f_{supp, inlet}(\theta, f) <p^2>^*_{\textrm{inlet}} \\
	    <p^2>^*_{\textrm{discharge, supp.}} & = \quad f_{supp, discharge}(\theta, f) <p^2>^*_{\textrm{discharge}} \\
	\end{aligned}
