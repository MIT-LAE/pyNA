.. _theory_file:

methods
=======

pyNA is developed based on the 1982 theoretical manuals from the Aircraft NOise Prediction Program (ANOPP) by Zorumski [Zorumski-1982]_. The following methods from literature are used in the different pyNA modules: 

noise source modules
--------------------

================================================ =============================================================================
Noise source module   							 Method
================================================ =============================================================================
Fan broadband and tones (inlet and discharge)    Heidmann method NASA TM X-71763 [Heidman-1975]_; 
												 * with GEAE revision NASA CR-195480 for BB [Kontos-1996a]_, 
												 * with AlliedSignal revision for RS tones [Hough-1996]_, 
												 * with fan treatment NASA CR-202309 [Kontos-1996b]_.
Combustor  									     Emmerling method FAA-RD-74-125 [Emmerling-1976]_. 
Jet mixing										 Single-stream, shock-free, circular jet mixing noise [SAE-2012]_.
Jet shock 										 Single-stream, circular jet shock cell noise (SAE ARP876-2012).
Airframe  										 Fink method FAA RD-77-29 [Fink-1977]_;
												 * with HSR calibration NASA CR-2004-213014 [Rawls-2004]_.
================================================ =============================================================================


noise propagation modules
-------------------------

================================================ =============================================================================
Noise propagation module   						 Method
================================================ =============================================================================
Spherical spreading  							 R-squared law 
Characteristic impedance effect 				 Characteristic impedance ratio.
Atmospheric absorption                           Exponential decay from source based on absorption coefficient.
Ground reflection and attenuation                Chien-Soroka method [Chien-1975]_.
Lateral attenuation and engine installation 	 Berton method [Berton-2021]_. 	
Wing shielding module                            Maekawa method [Maekawa-1968]_.
================================================ =============================================================================


noise levels modules
--------------------

================================================ ==============================================================================
Noise level module    							 Method
================================================ ==============================================================================
Perceived noise level, tone corrected (PNLT)     ICAO Annex 16 Volume I: Aircraft noise App. 2-13 [ICAO-2017]_.
Effective perceived noise level (EPNL) 		     ICAO Annex 16 Volume I: Aircraft noise App. 2-13 [ICAO-2017]_.	
================================================ ==============================================================================

References
----------

.. [Heidman-1975] Heidmann, M.F., *Interim prediction method for fan and compressor source noise (NASA-TM-X71763)*, 1975.
.. [Kontos-1996a] Kontos, K.B., Janardan,B., and Gliebe,P., *Improved NASA-ANOPP noise prediction computer code for advanced subsonic propulsion systems. Volume 1: ANOPP Evaluation and Fan Noise Model Improvement (NASA CR-195480)*, 1996.
.. [Hough-1996] Hough, J.W., and Weir, D.S., *Aircraft noise prediction program (ANOPP) fan noise prediction for small engines (NASA-CR- 198300)*, 1996.
.. [Kontos-1996b] Kontos, K. B., Kraft, R. E., and Gliebe, P. R., *Improved NASA-ANOPP Noise Prediction Computer Code for Advanced Subsonic Propulsion Systems. Volume 2: Fan Suppression Model Development (NASA-CR-202309)*, 1996.
.. [Emmerling-1976] Emmerling, J., Kazin, S., and Matta, R., *Core Engine Noise Program. Volume III. Prediction Methods–Supplement I.-Extension of Prediction Methods*, Tech. rep., General Electric Co Cincinnati OH Aircraft Engine Business Group, 1976.
.. [SAE-2012] Society of Automotive Engineers, *ARP-876: Gas Turbine Jet Exhaust Noise Prediction*, SAE, 1994.
.. [Fink-1977] Fink, M.R., *Airframe noise prediction method*, Technical report, United Technologies Research Center East Hartford, CT, 1977.
.. [Rawls-2004] Rawls, J.W., Yeager, J.C., *High Speed Research Noise Prediction Code (HSRNOISE) User's and Theoretical Manual (NASA CR-2004-213014)*, 2004.
.. [Chien-1975] Chien, C., and Soroka, W., *Sound propagation along an impedance plane*, Journal of Sound and Vibration, Vol.43, No.1, 1975, pp. 9–20.
.. [Maekawa-1968] Maekawa, Z., *Noise reduction by screens*, Applied Acoustics, Vol.1, No.3, 1968, pp.157–173.
.. [ICAO-2017] Annex 16 to the Convention on International Civil Aviation Environmental Protection Volume I: Aircraft Noise*, (2017). 
.. [Zorumski-1982] Zorumski, William E. *Aircraft noise prediction program theoretical manual, part 1-2.*, (1982).
.. [Berton-2021] Berton. *Simultaneous use of Ground Reflection and Lateral Attenuation Noise Models*. (2021).



