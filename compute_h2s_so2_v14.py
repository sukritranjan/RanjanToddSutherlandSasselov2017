# -*- coding: iso-8859-1 -*-
"""
Written by: Sukrit Ranjan, Zoe Todd

This script has three purposes:
PURPOSE 1: To compute the speciation of sulfur-bearing molecules from the dissolving of H2S (H2S, HS-, S--) and SO2 (SO2, HSO3-, SO3--, HS2O5-) in water, as a function of pH2S and pSO2.

The source is assumed to be the atmosphere, and the concentrations of [H2S] and [SO2] are computed via Henry's Law. Our results will always be valid near the air-water interface. They will be valid throughout our aqueous reservoir under the assumption that the reservoir is well-mixed. This assumption typically implies a shallow reservoir (i.e. lake, not ocean). 

Based on sensitivity studies, we do not include temperature effects in Henry's Law as they do not substantively affect our conclusions. We do, however, include salinity effects. We approximate the salinity by NaCl salinity. Note that in this limit, the ionic strength I=0.5*([Na]*1**2+[Cl]*1**2)=[Na]=[Cl].

Once we have calculated [H2S] ([SO2]), we must calculate subsequent speciation. The relevant reactions and rate constants we have identified are as follows:

H2S:
(1) H2S -------------> HS(-) + H(+)   pKa_1=7.05    1st dissociation     CRC Handbook, 90th Ed, p 8-40, Dissoc. Const. of Inorganics
(2) HS(-) -----------> S(2-) + H(+)   pKa_2=19      2nd dissociation     CRC Handbook, 90th Ed, p 8-40, Dissociation Constants of Inorganics

SO2:
At present, the method used is to consider the following reactions:
(1) SO2 + H2O -------> H(+) + HSO3(-) pKa_1=1.86         Bisulfite production       (Neta and Huie 1985)
(2) HSO3(-) ---------> H(+) + SO3(2-) pKa_2=7.2          Sulfate production         (Neta and Huie 1985)
(3) HSO3(-) + SO2 ---> HS2O5(-)       pKa_2=1.5          Disulfite production       (Neta and Huie 1985)

There are two ways to calculate subsequent speciation

METHOD 1: SET pH (Use activity-corrected Henderson-Hasselbach Equation)
In this formalism, we assume the aqueous reservoir is buffered by some other agent to a fixed pH. We do not specify this agent, so we cannot self-consistently include its reactions (in particular, there are problems with the equation of charge balance). We can, however, use the definition of Ka for an acid dissociation reaction:

For a reaction HA---> H+ + A- with dissociation constant Ka,
Ka=A_{H+}A_{A-}/A_{HA}
But, we know A_{H+} from the pH. Further, for the first dissociation we know A_{HA} from our Henry's Law calculation. So, we can solve this equation for the first dissociation, and then iterate for the rest of the dissociations.



METHOD 2: SOLVE SELF-CONSISTENT, UNBUFFERED SYSTEM (need to choose ionic strength I).
In this formalism, we assume that the sulfur speciation reactions are the _only_ reactions. In this case these processes set all parameters, including pH. To execute this calculation, we combine the dissociation equations for each species (above), with the concentration of H2S (SO2) set from Henry's Law, and with the water dissociation equation:

H2O -------------> H(+) + OH(-)   pKa_w=14           From a discussion with Amit

And the statement of charge conservation:

0=sum(all ionic species' concentrations)      Statement of charge conservation

This constitutes a system of N equations in N variables, that we can solve.



PURPOSE 2: To compute the speciation of sulfur-bearing molecules from the dissolving of H2S (H2S, HS-, S--) and SO2 (SO2, HSO3-, SO3--, HS2O5-) in buffered aqueous solution, as a function of planetary S-flux, using Hu et al. (2013) to connect the S-flux to pH2S, pSO2. The machinery from Purpose 1 is used to accomplish this.

PURPOSE 3: To plot the UV surface radiance as a function of total S-flux, using Hu et al. (2013) to connect the S-flux to pH2S, pSO2, and H2SO4 and S8 aerosol abundances, and our RT calculations to do the rest. The RT calculations are run from the RT folder via run_radiativetransfer.py
"""

########################
###Control Switches
########################
run_ph2s_pso2_chem=False #If true, runs pH2s, pSO2 calculations (Purpose 1). If False, does not.
run_s_flux_chem=False #If true, calculates everything as funciton of S-flux for buffered reservoir. (Purpose 2)
run_rt_plot=True #If true, plots the RT (Purpose 3)

atmtype='co2rich' # IF 'co2rich', runs the S-flux calculations for the CO2-rich case of Hu+2013. If 'n2rich', runs for the N2-rich case.

########################################################################
########################################################################
########################################################################
###SETUP & INPUTS
########################################################################
########################################################################
########################################################################

########################
###Import useful libraries
########################
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pdb

########################
###Define useful constants, all in CGS
########################

#Unit conversions
bar2atm=0.9869 #1 bar in atm

########################
###Henry's Law Parameters (from Burkholder et al 2015, the JPL compendium)
########################

###Reference temperature for all these parameters (room temp)
T_0_jpl=298.15 #reference temperature in K

###Henry's Law parameters for H2S
H_0_h2s=0.102*bar2atm #Henry Law Constant at 298K, from table 5-4 of Burkholder+2015, page 5-156. Units: M/atm converted to M/bar
h_g_0_h2s=-0.0333 #From table 5-4 of Burkholder+2015, page 5-156. Units: M**-1
h_t_h2s=0. #NOT FOUND FOR H2S, assumed to be 0.

###Henry's Law parameters for SO2
H_0_so2=1.36*bar2atm #Henry Law Constant at 298K, from table 5-4 of Burkholder+2015, page 5-156. Units: M/atm converted to M/bar
h_g_0_so2=-0.0607 #From table 5-4 of Burkholder+2015, page 5-156. Units: M**-1
h_t_so2=0.000275 #From table 5-4 of Burkholder+2015, page 5-156. Units: M**-1 K**-1

###Ion parameters, for salinity dependence of Henry's Law
h_i_na=0.1143 #For Na+. From Table 5-5 of Burkholder+2015, page 5-187
h_i_cl=0.0318 #For Cl-. From Table 5-5 of Burkholder+2015, page 5-187

########################
###For activity coefficient calculations
########################

A=0.5085 #Units: M^{-1/2}; valid for 25C
B=0.3281 #Units: M^{-1/2} A^{-1}; valid for 25C

########################
###Reaction pKas
########################

###Specify H2S speciation pKas
pKa_h2s_1=7.05 # From CRC Handbook, 90th Ed, p 8-40, Dissoc. Const. of Inorganics, for eqn H2S ------> HS(-) + H(+) 
pKa_h2s_2=19.  # From CRC Handbook, 90th Ed, p 8-40, Dissoc. Const. of Inorganics, for eqn H2(-)-----> S(2-) + H(+) 

###Specify SO2 speciation pKas
pKa_so2_1=1.86 #From Neta and Huie (1985, Environmental Health Perspectives, vol. 64). For SO2+H2O---->HSO3- + H+
pKa_so2_2=7.2 #From Neta and Huie (1985, Environmental Health Perspectives, vol. 64). For HSO3 ----->SO3(2-) + H+
pKa_so2_3=1.5 #From Neta and Huie (1985, Environmental Health Perspectives, vol. 64). For HSO3(-) + SO2 ---> HS2O5(-)

###Specify general pKas
# (w) H2O--->OH- + H+
pKa_w=14. #pKa for water dissociation

filelabel='' #No file label required for regular run (it is here to facilitate the temp-dependence sensitivity test.
###########################
######Temperature sensitivity testing; should normally be commented out
###########################
###########0C
########H_0_h2s=0.196 # Units: M/bar
########H_0_so2=3.36 # Units: M/bar
########A=0.4883 #Units: M^{-1/2}
########B=0.3241 #Units: M^{-1/2} A^{-1}
########pKa_h2s_1=7.098
########pKa_h2s_2=19.81
########pKa_so2_1=1.160
########pKa_so2_2=7.051
########filelabel='_0C'


##########50C
#######H_0_h2s=0.060 # Units: M/atm converted to M/bar
#######H_0_so2=0.647 # Units: M/bar
#######A=0.5319 #Units: M^{-1/2}; valid for 50C
#######B=0.3321 #Units: M^{-1/2} A^{-1}; valid for 50C
#######pKa_h2s_1=7.009
#######pKa_h2s_2=18.31
#######pKa_so2_1=2.452
#######pKa_so2_2=7.326
#######filelabel='_50C'


########################
###Convert pKas to Kas
########################
Ka_h2s_1=10.**(-pKa_h2s_1)
Ka_h2s_2=10.**(-pKa_h2s_2)

Ka_so2_1=10.**(-pKa_so2_1)
Ka_so2_2=10.**(-pKa_so2_2)
Ka_so2_3=10.**(-pKa_so2_3)

Ka_w=10.**(-pKa_w)

########################
###Hu et al. 2013 mapping
########################


###From Hu+2013:
phi_s_list=np.array([3.e9, 1.e10, 3.e10, 1.e11, 3.e11, 1.e12]) #units: cm**-2 s**-1
phi_s_list_labels=np.array(['3.e9', '1.e10', '3.e10', '1.e11', '3.e11', '1.e12']) #for legend

if atmtype=='co2rich':
	mr_h2s_list=np.array([4.e-10, 1.e-9, 9.e-9, 5.e-8, 2.e-7, 7.e-7]) #column-integrated mixing ratios
	mr_so2_list=np.array([3.e-10, 9.e-10, 3.e-9, 7.e-9, 1.e-8, 3.e-8]) #column-integrated mixing ratios
elif atmtype=='n2rich':
	mr_h2s_list=np.array([9.e-14, 9.e-14, 2.e-13, 7.e-13, 4.e-10, 1.e-7]) #column-integrated mixing ratios; first 2 are upper bounds
	mr_so2_list=np.array([4.e-12, 1.e-11, 5.e-11, 2.e-10, 2.e-9, 2.e-8]) #column-integrated mixing ratios
else:
	print 'Error: incorrect value for variable atmtype'

########################
###Chemistry Sub Functions: Henry's Law, activity coefficients
########################
def get_aqueous_concentration(P_gas, H):
	"""
	Function implementing Henry's Law.
	Takes: Gas partial pressure (in bar), Henry's Law Constant (in M/bar)
	Returns: dissolved concentration in M
	Fundamental equation: [X]_aq=H*P_gas
	"""
	return P_gas*H

def correct_H_NaCl_salinity_jpl(H, h_g_0, h_t, T, conc_NaCl):
	"""
	This function calculates the Henry's Law constant as a function of salinity. It assumes NaCl is the SOLE source of salinity.
	
	Takes: Henry's Law constant, temperature-independent gas-specific Sechenov coefficient, temperature-dependent gas-specific Sechenov coefficient, temperature (K), [NaCl] (M)
	Returns: corrected Henry's Law constant
	Uses: Sechenov equations, see p 5-187 of Burkholder et al 2015 for example
	"""
	h_g=h_g_0+h_t*(T-T_0_jpl) #all temperature dependence of sechenov constant embedded here, in calculating the gas-specific parameter
	
	K_NaCl=(h_i_na+h_g)*(1.) + (h_i_cl+h_g)*(1.) #I interpret ion index based on the sample calculation of pg 5-187...
	
	####Test case. From Worsnop+1995, K_NaCl for H2S at 278 K is 0.064 M**-1, with values in the literature ranging from 0.741-0.0811. With our numbers, we get K_NaCl for H2S of 0.0795. Not bad!
	
	H_corr=H*10.**(-K_NaCl*conc_NaCl)
	
	return H_corr

def activity_coefficient(species, I):
	"""
	Function encoding the activity coefficient for the 5 species in our system
	Takes: species name, ionic strength (M)
	Returns: activity coefficient in 1/M
	Fundamental equation: A_x=C_x*[x], where A_x is the activity of x and C_x is the associated activity coefficient of x.
	Two regimes used for this calculation: Below I=0.1M, Extended Debye-Huckel theory is used. Above I=0.1M, Truesdell-Jones theory is used. This is valid up to 1M. 
	"""
	if species=='H+':
		ai=9.0 #units: A
		z=1.0
		logcoeff=(-A*z*z*np.sqrt(I))/(1.+ai*B*np.sqrt(I))
	elif species=='OH-':
		ai=3.5 #Units: A
		z=1.0
		logcoeff=(-A*z*z*np.sqrt(I))/(1.+ai*B*np.sqrt(I))
	elif species=='H2S':
		logcoeff=0.
	elif species=='HS-':
		ai=3.5 #unit: A
		z=1.0
		logcoeff=(-A*z*z*np.sqrt(I))/(1.+ai*B*np.sqrt(I))
	elif species=='S--':
		ai=5.0 #Unit: A
		z=2.0
		logcoeff=(-A*z*z*np.sqrt(I))/(1.+ai*B*np.sqrt(I))
	elif species=='SO2':
		logcoeff=0.
	elif species=='HSO3-':
		ai=4.0 #Units: A
		z=1.0
		logcoeff=(-A*z*z*np.sqrt(I))/(1.+ai*B*np.sqrt(I))
	elif species=='SO3--':
		ai=4.5 #Units:A
		z=2.0
		logcoeff=(-A*z*z*np.sqrt(I))/(1.+ai*B*np.sqrt(I))
	elif species=='HS2O5-':
		logcoeff=0. #We lack the ai for this, so neglect its activity coefficient calculation
	else:
		print('error: incorrect input in function activity_coefficient')
		return False
	coeff=10.**logcoeff
	return coeff

########################################################################
########################################################################
########################################################################
###DEFINE FUNCTIONS TO IMPLEMENT CALCULATIONS
########################################################################
########################################################################
########################################################################

########################
###Subfunctions: H2S, SO2 Speciation in Buffered Solution (Method 1)
########################

def solve_buffered_h2s(conc_h2s, pH, I):
	"""
	Function computes speciation of H2S in aqueous solution assuming external factors are setting the pH (e.g., calcium carbonate buffering)
	Takes: [H2S] (M), pH, ionic strength (M)
	Returns: ([H+],[OH-], [HS-], [S--]), all in M
	Uses: definition of Ka
	"""
	#definition of pH
	a_h=10.**(-pH) #activity of H+
	a_oh=Ka_w/a_h #activity of OH-
	
	conc_h=a_h/activity_coefficient('H+',I) #extract concentration
	conc_oh=a_oh/activity_coefficient('OH-',I) #extract concentration
	
	#First speciation
	a_h2s=activity_coefficient('H2S',I)*conc_h2s #activity of H2S
	a_hs=Ka_h2s_1*a_h2s/a_h #activity of HS-
	conc_hs=a_hs/activity_coefficient('HS-', I)#concentration of HS-
	
	#Second speciation
	a_s=Ka_h2s_2*a_hs/a_h #activity of S--
	conc_s=a_s/activity_coefficient('S--', I)

	result=(conc_h, conc_oh, conc_hs, conc_s)
	return result

def solve_buffered_so2(conc_so2, pH, I):
	"""
	Function computes speciation of SO2 in aqueous solution assuming external factors are setting the pH (e.g., calcium carbonate buffering)
	Takes: [SO2] (M), pH, ionic strength (M)
	Returns: ([H+],[OH-], [HS-], [S--]), all in M
	Uses: definition of Ka
	"""
	#definition of pH
	a_h=10.**(-pH) #activity of H+
	a_oh=Ka_w/a_h #activity of OH-
	
	conc_h=a_h/activity_coefficient('H+',I) #extract concentration
	conc_oh=a_oh/activity_coefficient('OH-',I) #extract concentration
	
	#First speciation
	a_so2=activity_coefficient('SO2',I)*conc_so2 #activity of SO2
	a_hso3=Ka_so2_1*a_so2/a_h #activity of HS-
	conc_hso3=a_hso3/activity_coefficient('HSO3-', I)#concentration of HS-
	
	#Second speciation
	a_so3=Ka_so2_2*a_hso3/a_h #activity of SO3--
	conc_so3=a_so3/activity_coefficient('SO3--', I)
	
	#Third speciation
	a_hs2o5=Ka_so2_3*a_hso3*a_so2 #activity of HS2O5-
	conc_hs2o5=a_hs2o5/activity_coefficient('HS2O5-', I)

	result=(conc_h, conc_oh, conc_hso3, conc_so3, conc_hs2o5)
	return result

########################
###Subfunctions: H2S Equilibrium Chemistry Calculation (Method 2, Unbuffered System)
########################
def system_h2s(p, ln_conc_h2s, I):
	"""
	Implements system of equations for fsolve to solve, with [H2S] taken as a given.
	Uses as a model http://stackoverflow.com/questions/8739227/how-to-solve-a-pair-of-nonlinear-equations-using-python

	Uses log-version of equations to make tractable for numerical solver
	"""
	ln_conc_h, ln_conc_oh, ln_conc_hs, ln_conc_s = p #extract concentrations from input vector

	#extract non-log values, for charge conservation step (unfortunately)
	conc_h=np.exp(ln_conc_h)
	conc_oh=np.exp(ln_conc_oh)
	conc_hs=np.exp(ln_conc_hs)
	conc_s=np.exp(ln_conc_s)

	#convert to activities
	ln_a_h2s=np.log(activity_coefficient('H2S', I))+ln_conc_h2s
	ln_a_h=np.log(activity_coefficient('H+', I))+ln_conc_h
	ln_a_oh=np.log(activity_coefficient('OH-', I))+ln_conc_oh
	ln_a_hs=np.log(activity_coefficient('HS-', I))+ln_conc_hs
	ln_a_s=np.log(activity_coefficient('S--', I))+ln_conc_s

	eqn1 = np.log(Ka_h2s_1)-(ln_a_hs+ln_a_h-ln_a_h2s)
	eqn2 = np.log(Ka_h2s_2)-(ln_a_s+ln_a_h-ln_a_hs)
	eqn3 = np.log(Ka_w)-(ln_a_h+ln_a_oh)
	eqn4 = conc_h-2.*conc_s-conc_hs-conc_oh

	return (eqn1, eqn2, eqn3, eqn4)

def system_h2s_nonlog_residuals(p, conc_h2s, I):
	"""
	prints fractional residuals of system of equations to verify precision to which root has been found, WITHOUT logging for numerical stability. This lets us check the solution we have obtained.
	"""
	conc_h, conc_oh, conc_hs, conc_s= p #extract concentrations from input vector

	#convert to activities
	a_h2s=activity_coefficient('H2S', I)*conc_h2s
	a_h=activity_coefficient('H+', I)*conc_h
	a_oh=activity_coefficient('OH-', I)*conc_oh
	a_hs=activity_coefficient('HS-', I)*conc_hs
	a_s=activity_coefficient('S--', I)*conc_s

	eqn1 = (Ka_h2s_1-(a_hs*a_h/a_h2s))/Ka_h2s_1
	eqn2 = (Ka_h2s_2-(a_s*a_h/a_hs))/Ka_h2s_2
	eqn3 = (Ka_w-(a_h*a_oh))/Ka_w
	eqn4 = (conc_h-2.*conc_s-conc_hs-conc_oh)/(conc_h+2.*conc_s+conc_hs+conc_oh)

	return (eqn1, eqn2, eqn3, eqn4)


def solve_unbuffered_h2s(conc_h2s, I):
	"""
	Function numerically solving the system articulated in the header, with [H2S] taken as given from Henry's Law.
	Takes: [H2S] (M), Ionic strength (M)
	Returns: ([H+],[OH-], [HS-], [S--]), all in M
	Uses: scipy.optimize.fsolve, a multivariate root-finder

	Programatic finders have trouble with systems that have orders-of-magnitude varying quantities. Therefore, we solve for the ln of the concentrations, and then exponentiate to recover the true concentrations
	"""
	ln_conc_h2s=np.log(conc_h2s)

	starting_guesses=(np.log(1.e-7), np.log(1.e-7), ln_conc_h2s, ln_conc_h2s) #initial guesses for roots. Need to be right scale. 
	solution = scipy.optimize.fsolve(system_h2s, starting_guesses, args=(ln_conc_h2s, I), xtol=1.e-6) #solve system to 1.e-6 relative precision 

	conc_h=np.exp(solution[0])
	conc_oh=np.exp(solution[1])
	conc_hs=np.exp(solution[2])
	conc_s=np.exp(solution[3])
	result=(conc_h, conc_oh, conc_hs, conc_s)

	#print np.max(np.abs(system_h2s_nonlog_residuals(result, conc_h2s, I))) #check that all residuals are 0 within required precision, signifying that system has successfully solved.
	return result

########################
###Subfunctions: SO2 Equilibrium Chemistry Calculation (Method 2, Unbuffered System)
########################
def system_so2(p, ln_conc_so2, I):
	"""
	Implements system of equations for fsolve to solve, with [SO2] taken as a given.
	Uses as a model http://stackoverflow.com/questions/8739227/how-to-solve-a-pair-of-nonlinear-equations-using-python
	
	Uses log-version of equations to make tractable for numerical solver
	"""
	ln_conc_h, ln_conc_oh, ln_conc_hso3, ln_conc_so3, ln_conc_hs2o5 = p #extract concentrations from input vector
	
	#extract non-log values, for charge conservation step (unfortunately)
	conc_h=np.exp(ln_conc_h)
	conc_oh=np.exp(ln_conc_oh)
	conc_hso3=np.exp(ln_conc_hso3)
	conc_so3=np.exp(ln_conc_so3)
	conc_hs2o5=np.exp(ln_conc_hs2o5)
	
	#convert to activities
	ln_a_so2=np.log(activity_coefficient('SO2',I))+ln_conc_so2
	ln_a_h=np.log(activity_coefficient('H+',I))+ln_conc_h
	ln_a_oh=np.log(activity_coefficient('OH-',I))+ln_conc_oh
	ln_a_hso3=np.log(activity_coefficient('HSO3-',I))+ln_conc_hso3
	ln_a_so3=np.log(activity_coefficient('SO3--',I))+ln_conc_so3
	ln_a_hs2o5=np.log(activity_coefficient('HS2O5-',I))+ln_conc_hs2o5
	
	eqn1 = np.log(Ka_so2_1)-(ln_a_hso3+ln_a_h-ln_a_so2)
	eqn2 = np.log(Ka_so2_2)-(ln_a_so3+ln_a_h-ln_a_hso3)
	eqn3 = np.log(Ka_so2_3)-(ln_a_hs2o5-ln_a_hso3-ln_a_so2)
	eqn4 = np.log(Ka_w)-(ln_a_h+ln_a_oh)
	eqn5 = conc_h-2.*conc_so3-conc_hso3-conc_oh-conc_hs2o5
	
	return (eqn1, eqn2, eqn3, eqn4, eqn5)

def system_so2_nonlog_residuals(p, conc_so2, I):
	"""
	prints fractional residuals of system of equations to verify precision to which root has been found, WITHOUT logging for numerical stability. This lets us check the solution we have obtained.
	"""
	conc_h, conc_oh, conc_hso3, conc_so3, conc_hs2o5= p #extract concentrations from input vector
	
	#convert to activities
	a_so2=activity_coefficient('SO2',I)*conc_so2
	a_h=activity_coefficient('H+',I)*conc_h
	a_oh=activity_coefficient('OH-',I)*conc_oh
	a_hso3=activity_coefficient('HSO3-',I)*conc_hso3
	a_so3=activity_coefficient('SO3--',I)*conc_so3
	a_hs2o5=activity_coefficient('HS2O5-',I)*conc_hs2o5
	
	eqn1 = (Ka_so2_1-(a_hso3*a_h/a_so2))/Ka_so2_1
	eqn2 = (Ka_so2_2-(a_so3*a_h/a_hso3))/Ka_so2_2
	eqn3 = (Ka_so2_3-(a_hs2o5/(a_hso3*a_so2)))/Ka_so2_3
	eqn4 = (Ka_w-(a_h*a_oh))/Ka_w
	eqn5 = (conc_h-2.*conc_so3-conc_hso3-conc_oh-conc_hs2o5)/(conc_h+2.*conc_so3+conc_hso3+conc_oh+conc_hs2o5) #residual in M of charge
	
	return (eqn1, eqn2, eqn3, eqn4, eqn5)

def solve_unbuffered_so2(conc_so2,I):
	"""
	Function numerically solving the system articulated in the header, with [SO2] taken as given from Henry's Law.
	Takes: [SO2], in M, I=ionic strength
	Returns: [[H(+)],[OH(-)], [HSO3(-)], [SO3(2-)], [HS2O5(-)]], all in M
	Uses: scipy.optimize.fsolve, a multivariate root-finder
	
	Programatic finders have trouble with systems that have orders-of-magnitude varying quantities. Therefore, we solve for the ln of the concentrations, and then exponentiate to recover the true concentrations
	"""
	ln_conc_so2=np.log(conc_so2)
	
	starting_guesses=(np.log(1.e-7), np.log(1.e-7), ln_conc_so2, ln_conc_so2, ln_conc_so2) #initial guesses for roots. Need to be right scale.
	solution = scipy.optimize.fsolve(system_so2, starting_guesses, args=(ln_conc_so2, I), xtol=1.e-6) #solve system to 1.e-6 relative precision

	conc_h=np.exp(solution[0])
	conc_oh=np.exp(solution[1])
	conc_hso3=np.exp(solution[2])
	conc_so3=np.exp(solution[3])
	conc_hs2o5=np.exp(solution[4])
	result=(conc_h, conc_oh, conc_hso3, conc_so3, conc_hs2o5)

	#print np.max(np.abs(system_so2_nonlog_residuals(result, conc_so2, I))) #check that all residuals are 0 within required precision, signifying that system has successfully solved.
	
	return result

if run_ph2s_pso2_chem:
	########################################################################
	########################################################################
	########################################################################
	###RUN CALCULATIONS: Exploring all chemistry work
	########################################################################
	########################################################################
	########################################################################

	########################
	###Define Inputs
	########################
	#define inputs
	p_h2s_list=10.**(np.arange(-12., -1.)) #H2S partial pressures to consider, in bar
	p_so2_list=10.**(np.arange(-12., -1.)) #SO2 partial pressures to consider, in bar

	pH_list=np.array([4.25, 7., 8.2]) #List of pHs we consider. 4.25 comes from Halevy et al (2007) estimate for lower bound on pH of raindrops in contact with 0.2 bar CO2; 7 corresponds to experimental conditions of Patel+2015, buffered by phospate; 8.2 corresponds to modern ocean.

	I_list=np.array([0., 0.1]) #List of ionic strengths to consider: 0 (no ions), 0.1 (approx limit from vesicle formation), and 1 (approx modern oceans, at 0.7). Upper bound on salinity comes from saturation of water; dead sea at ~7 is approx at this limit.

	I_0_ind=0 
	I_0=I_list[I_0_ind] #Choose 0 salinity because the salinity of lakes, etc is negligible from POV of activity coefficients.Similarly, I_rivers=.001 #OOM average river salinity, see Physics and Chemistry of Lakes, Second Edition, p 268, and references within.  See also: http://funnel.sfsu.edu/courses/geol480/lectures/lecture7.pdf. 

	########################
	###Initialize variables to hold calculation results.
	########################
	#array to hold [H2S], [SO2] concentrations as function of pSO2 and for different salinities
	conc_h2s_list_ph2s_I=np.zeros([len(p_h2s_list), len(I_list)])
	conc_so2_list_pso2_I=np.zeros([len(p_so2_list), len(I_list)])

	#array to hold concentrations for buffered system for H2S (Method 1)
	conc_h_list_buff_ph2s_pH=np.zeros([len(p_h2s_list), len(pH_list)])
	conc_oh_list_buff_ph2s_pH=np.zeros([len(p_h2s_list), len(pH_list)])
	conc_hs_list_buff_ph2s_pH=np.zeros([len(p_h2s_list), len(pH_list)])
	conc_s_list_buff_ph2s_pH=np.zeros([len(p_h2s_list), len(pH_list)])

	#array to hold concentrations for unbuffered system for H2S (Method 2)
	conc_h_list_unbuff_ph2s_I=np.zeros([len(p_h2s_list), len(I_list)])
	conc_oh_list_unbuff_ph2s_I=np.zeros([len(p_h2s_list), len(I_list)])
	conc_hs_list_unbuff_ph2s_I=np.zeros([len(p_h2s_list), len(I_list)])
	conc_s_list_unbuff_ph2s_I=np.zeros([len(p_h2s_list), len(I_list)])

	#array to hold concentrations for buffered system for SO2 (Method 1)
	conc_h_list_buff_pso2_pH=np.zeros([len(p_so2_list), len(pH_list)])
	conc_oh_list_buff_pso2_pH=np.zeros([len(p_so2_list), len(pH_list)])
	conc_hso3_list_buff_pso2_pH=np.zeros([len(p_so2_list), len(pH_list)])
	conc_so3_list_buff_pso2_pH=np.zeros([len(p_so2_list), len(pH_list)])
	conc_hs2o5_list_buff_pso2_pH=np.zeros([len(p_so2_list), len(pH_list)])

	#array to hold concentrations for unbuffered system for SO2 (Method 2)
	conc_h_list_unbuff_pso2_I=np.zeros([len(p_so2_list), len(I_list)])
	conc_oh_list_unbuff_pso2_I=np.zeros([len(p_so2_list), len(I_list)])
	conc_hso3_list_unbuff_pso2_I=np.zeros([len(p_so2_list), len(I_list)])
	conc_so3_list_unbuff_pso2_I=np.zeros([len(p_so2_list), len(I_list)])
	conc_hs2o5_list_unbuff_pso2_I=np.zeros([len(p_so2_list), len(I_list)])



	########################
	###Calculate [H2S], [SO2] from Henry's Law for different ionic strengths
	########################
	#Get [H2S] from Henry's Law, for different ionic strengths
	for ind in range(0, len(I_list)):
		I=I_list[ind] #ionic strength
		conc_NaCl=0.#I #for NaCl, the ionic strength is just 0.5*([Na]+[Cl])=[Na]=[Cl]=[NaCl]
		
		H_h2s=correct_H_NaCl_salinity_jpl(H_0_h2s, h_g_0_h2s, h_t_h2s, T_0_jpl, conc_NaCl) #ignore temperature dependence of salting out...second order effect
		conc_h2s_list_ph2s_I[:,ind]=get_aqueous_concentration(p_h2s_list, H_h2s) #concentration of dissolved H2S from Henry's Law, in M, for specified salinity (T=T_0)
		
		H_so2=correct_H_NaCl_salinity_jpl(H_0_so2, h_g_0_so2, h_t_so2, T_0_jpl, conc_NaCl) #ignore temperature dependence of salting out...second order effect
		conc_so2_list_pso2_I[:,ind]=get_aqueous_concentration(p_so2_list, H_so2) #concentration of dissolved H2S from Henry's Law, in M, for specified salinity (T=T_0)	

	########################
	###Calculate speciation for H2S
	########################

	for ph2s_ind in range(0, len(p_h2s_list)):
		###Buffered solution
		for pH_ind in range(0, len(pH_list)):
			pH=pH_list[pH_ind]

			conc_h2s=conc_h2s_list_ph2s_I[ph2s_ind, I_0_ind] #specify which of the Henry's Law calculations [f(salinity)] using for this calculation


			conc_h_list_buff_ph2s_pH[ph2s_ind, pH_ind], conc_oh_list_buff_ph2s_pH[ph2s_ind, pH_ind], conc_hs_list_buff_ph2s_pH[ph2s_ind, pH_ind], conc_s_list_buff_ph2s_pH[ph2s_ind, pH_ind] = solve_buffered_h2s(conc_h2s, pH, I_0)	
		
		###Unbuffered solution
		for I_ind in range(0,len(I_list)):
			I=I_list[I_ind]

			conc_h2s=conc_h2s_list_ph2s_I[ph2s_ind, I_ind] #specify which of the Henry's Law calculations [f(salinity)] using for this calculation
			
			conc_h_list_unbuff_ph2s_I[ph2s_ind, I_ind], conc_oh_list_unbuff_ph2s_I[ph2s_ind, I_ind], conc_hs_list_unbuff_ph2s_I[ph2s_ind, I_ind], conc_s_list_unbuff_ph2s_I[ph2s_ind, I_ind] = solve_unbuffered_h2s(conc_h2s, I)


	########################
	###Calculate speciation for SO2
	########################

	for pso2_ind in range(0, len(p_so2_list)):
		###Buffered solution
		for pH_ind in range(0, len(pH_list)):
			pH=pH_list[pH_ind]

			conc_so2=conc_so2_list_pso2_I[pso2_ind, I_0_ind] #specify which of the Henry's Law calculations [f(salinity)] using for this calculation

			conc_h_list_buff_pso2_pH[pso2_ind, pH_ind], conc_oh_list_buff_pso2_pH[pso2_ind, pH_ind], conc_hso3_list_buff_pso2_pH[pso2_ind, pH_ind], conc_so3_list_buff_pso2_pH[pso2_ind, pH_ind], conc_hs2o5_list_buff_pso2_pH[pso2_ind, pH_ind] = solve_buffered_so2(conc_so2, pH, I_0)	
		
		###Unbuffered solution
		for I_ind in range(0,len(I_list)):
			I=I_list[I_ind]

			conc_so2=conc_so2_list_pso2_I[pso2_ind, I_ind] #specify which of the Henry's Law calculations [f(salinity)] using for this calculation
			
			conc_h_list_unbuff_pso2_I[pso2_ind, I_ind], conc_oh_list_unbuff_pso2_I[pso2_ind, I_ind], conc_hso3_list_unbuff_pso2_I[pso2_ind, I_ind], conc_so3_list_unbuff_pso2_I[pso2_ind, I_ind], conc_hs2o5_list_unbuff_pso2_I[pso2_ind, I_ind] = solve_unbuffered_so2(conc_so2, I)
			
			

	########################################################################
	########################################################################
	########################################################################
	###PLOT RESULTS: Exploring all chemistry work
	########################################################################
	########################################################################
	########################################################################

	########################
	###Plot H2S Results
	########################
	fig, ax=plt.subplots(2,2, figsize=(11., 7.), sharex=True, sharey=False)
	markersizeval=5.
	colors=cm.rainbow(np.linspace(0,1,len(I_list)+len(pH_list)))

	ax[0,0].set_title('[H2S]')
	ax[1,0].set_title('[HS-]')
	ax[0,1].set_title('[S(2-)]')
	ax[1,1].set_title('pH')


	for pH_ind in range(0, len(pH_list)):
		pH=pH_list[pH_ind]

		ax[1,0].plot(p_h2s_list, conc_hs_list_buff_ph2s_pH[:, pH_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[pH_ind], label='Buffered, pH='+str(pH))
		ax[0,1].plot(p_h2s_list, conc_s_list_buff_ph2s_pH[:, pH_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[pH_ind], label='Buffered, pH='+str(pH))
		ax[1,1].plot(p_h2s_list, -np.log10(activity_coefficient('H+', I_0)*conc_h_list_buff_ph2s_pH[:, pH_ind]), marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[pH_ind], label='Buffered, pH='+str(pH))
		
		#pdb.set_trace()

	numpH=len(pH_list)

	for I_ind in range(0, len(I_list)):
		I=I_list[I_ind]
		
		ax[0,0].plot(p_h2s_list, conc_h2s_list_ph2s_I[:,I_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[I_ind+numpH])

		ax[1,0].plot(p_h2s_list, conc_hs_list_unbuff_ph2s_I[:, I_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[I_ind+numpH], label='Unbuffered, I='+str(I))
		ax[0,1].plot(p_h2s_list, conc_s_list_unbuff_ph2s_I[:, I_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[I_ind+numpH], label='Unbuffered, I='+str(I))
		ax[1,1].plot(p_h2s_list, -np.log10(activity_coefficient('H+', I)*conc_h_list_unbuff_ph2s_I[:,I_ind]), marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[I_ind+numpH], label='Unbuffered, I='+str(I))
		
	for ax1ind in range(0, 2):
		for ax2ind in range(0,2):
			#To guide the eye as to what is plausible: use models of Hu et al (2013). Specically, use the 90%CO2, 10%N2 atmosphere in Figure 5. Not a perfect recreation of the young Earth (too much H2S/too little SO2 emitted, too little N2, stellar spectral shape different), but best available guide. Probably on optimistic side.
			ax[ax1ind, ax2ind].axvspan(1.e-4, 1., color='pink', alpha=0.2) #Ranjan+2016b photochemical-quenching level

			ax[ax1ind, ax2ind].axvline(4.e-10, linestyle=':', color='black') #Level corresponding to S emission rate of 3e9 S cm**-2 s**-1 in the Hu et al atmosphere, upper end of modern outgassing/down the middle for early Earth.
			ax[ax1ind, ax2ind].axvline(2.e-7, linestyle='--', color='black') #Level corresponding to S emission rate of 3e11 S cm**-2 s**-1 in the Hu et al atmosphere. S emission rate comes from Halevy & Head 2014, upper range of what is possible from terrestrial flood basalts
			
			
			if not((ax1ind==1) and (ax2ind==1)): #exclude the pH plot
				ax[ax1ind, ax2ind].axhline(1.e-3, linestyle=':', color='black') #demarcating millimolar concentrations
				ax[ax1ind, ax2ind].axhline(1.e-6, linestyle='--', color='black') #demarcating micromolar concentrations


	#ax[0,0].set_xlabel('pH2S (bar)')
	ax[0,0].set_xscale('log')
	ax[0,0].set_ylabel('[H2S] (M)')
	ax[0,0].set_yscale('log')

	ax[1,0].set_xlabel('pH2S (bar)')
	ax[1,0].set_xscale('log')
	ax[1,0].set_ylabel('[HS-] (M)')
	ax[1,0].set_yscale('log')

	ax[0,1].set_xlabel('pH2S (bar)')
	ax[0,1].set_xscale('log')
	ax[0,1].set_ylabel('[S(2-)] (M)')
	ax[0,1].set_yscale('log')

	ax[1,1].set_xlim([np.min(p_h2s_list), np.max(p_h2s_list)])
	ax[1,1].set_xlabel('pH2S (bar)')
	ax[1,1].set_xscale('log')
	ax[1,1].set_ylabel('pH')
	ax[1,1].set_yscale('linear')


	ax[0,1].legend(bbox_to_anchor=[-0.63, 1.05, 2., .152], loc=3, ncol=2, borderaxespad=0., fontsize=12)
	plt.tight_layout(rect=(0,0,1,0.92))

	plt.savefig('./Plots/h2s_speciation_paper'+filelabel+'.eps', orientation='portrait',papertype='letter', format='eps')

	########################
	###Plot SO2 Results
	########################
	fig, ax=plt.subplots(3,2, figsize=(11., 7.), sharex=True, sharey=False)
	markersizeval=5.
	colors=cm.rainbow(np.linspace(0,1,len(I_list)+len(pH_list)))

	ax[0,0].set_title('[SO2]')
	ax[1,0].set_title('[HSO3-]')
	ax[2,0].set_title('[SO3(2-)]')
	ax[1,1].set_title('[HS2O5(-)]')
	ax[2,1].set_title('pH')

	for pH_ind in range(0, len(pH_list)):
		pH=pH_list[pH_ind]
		ax[1,0].plot(p_so2_list, conc_hso3_list_buff_pso2_pH[:, pH_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[pH_ind], label='Buffered, pH='+str(pH))
		ax[2,0].plot(p_so2_list, conc_so3_list_buff_pso2_pH[:, pH_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[pH_ind], label='Buffered, pH='+str(pH))
		ax[1,1].plot(p_so2_list, conc_hs2o5_list_buff_pso2_pH[:, pH_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[pH_ind], label='Buffered, pH='+str(pH))	
		ax[2,1].plot(p_so2_list, -np.log10(activity_coefficient('H+', I_0)*conc_h_list_buff_pso2_pH[:, pH_ind]), marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[pH_ind], label='Buffered, pH='+str(pH))

	numpH=len(pH_list)

	for I_ind in range(0, len(I_list)):
		I=I_list[I_ind]

		ax[0,0].plot(p_so2_list, conc_so2_list_pso2_I[:,I_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[I_ind+numpH])

		ax[1,0].plot(p_so2_list, conc_hso3_list_unbuff_pso2_I[:, I_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[I_ind+numpH], label='Unbuffered, I='+str(I))
		ax[2,0].plot(p_so2_list, conc_so3_list_unbuff_pso2_I[:, I_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[I_ind+numpH], label='Unbuffered, I='+str(I))
		ax[1,1].plot(p_so2_list, conc_hs2o5_list_unbuff_pso2_I[:, I_ind], marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[I_ind+numpH], label='Unbuffered, I='+str(I))
		ax[2,1].plot(p_so2_list, -np.log10(activity_coefficient('H+', I)*conc_h_list_unbuff_pso2_I[:, I_ind]), marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[I_ind+numpH], label='Unbuffered, I='+str(I))

	for ax1ind in range(0, 3):
		for ax2ind in range(0,2):
			#To guide the eye as to what is plausible: use models of Hu et al (2013). Specically, use the 90%CO2, 10%N2 atmosphere in Figure 5. Not a perfect recreation of the young Earth (too much H2S/too little SO2 emitted, too little N2, stellar spectral shape different), but not a bad guide. Probably on optimistic side.
			ax[ax1ind, ax2ind].axvspan(1.e-5, 1., color='pink', alpha=0.2) #Ranjan+2016b photochemical-quenching level
			
			ax[ax1ind, ax2ind].axvline(3.e-10, linestyle=':', color='black') #Level corresponding to S emission rate of 1e11.5 S cm**-2 s**-1 in the Hu et al atmosphere. S emission rate comes from Halevy & Head 2014, upper range of what is possible from terrestrial flood basalts
			ax[ax1ind, ax2ind].axvline(1.e-8, linestyle='--', color='black') #Level corresponding to S emission rate of 1e11.5 S cm**-2 s**-1 in the Hu et al atmosphere. S emission rate comes from Halevy & Head 2014, upper range of what is possible from terrestrial flood basalts
			
			
			if not((ax1ind==2) and (ax2ind==1)): #exclude the pH plot
				ax[ax1ind, ax2ind].axhline(1.e-3, linestyle=':', color='black') #demarcating millimolar concentrations
				ax[ax1ind, ax2ind].axhline(1.e-6, linestyle='--', color='black') #demarcating micromolar concentrations

	ax[0,0].set_xscale('log')
	ax[0,0].set_xlim([np.min(p_so2_list), np.max(p_so2_list)])
	ax[0,0].set_ylabel('[SO2] (M)')
	ax[0,0].set_yscale('log')

	ax[1,0].set_xscale('log')
	ax[1,0].set_ylabel('[HSO3-] (M)')
	ax[1,0].set_yscale('log')

	ax[2,0].set_xscale('log')
	ax[2,0].set_ylabel('[SO3(2-)] (M)')
	ax[2,0].set_yscale('log')
	ax[2,0].set_xlabel('pSO2 (bar)')

	ax[1,1].set_xscale('log')
	ax[1,1].set_ylabel('HS2O5(-)')
	ax[1,1].set_yscale('log')
	ax[1,1].set_yscale('log')
	ax[1,1].set_xticks(p_so2_list)
	ax[1,1].set_yticks(10.**np.arange(-23, 3, step=3))


	ax[2,1].set_xlabel('pSO2 (bar)')
	ax[2,1].set_xscale('log')
	ax[2,1].set_ylabel('pH')
	ax[2,1].set_yscale('linear')
	#ax[2,0].set_xlim([np.min(p_so2_list), np.max(p_so2_list)])


	fig.delaxes(ax[0,1])
	ax[1,1].legend(bbox_to_anchor=[-0.1, 1.5, 2., .152], loc=3, ncol=2, borderaxespad=0., fontsize=12)
	plt.tight_layout(rect=(0,0,1,1))

	plt.savefig('./Plots/so2_speciation_paper'+filelabel+'.eps', orientation='portrait',papertype='letter', format='eps')


if run_s_flux_chem:
	#########################################################################
	#########################################################################
	#########################################################################
	####RUN CALCULATIONS: Explore H2S and SO2 together as function of phi_S using Hu et al (2013) as guide
	#########################################################################
	#########################################################################
	#########################################################################

	#########################
	####Establish key inputs
	#########################
	###Convert mixing ratios to surface pressures. Hu et al assume 1 bar atmosphere. We approximate the surface mixing ratios by the column-integrated mixing ratios they provide; surface-integrated mixing ratios > column-integrated mixing ratios, so these levels should be taken as lower bound.
	p_h2s_list=1.*mr_h2s_list #bar
	p_so2_list=1.*mr_so2_list #bar

	###Conduct calculations at pH=7 (approximately what the Patel et al 2015 pathways were conducted in, buffered by phospate) and an ionic strength of 0
	pH=7.
	I=0.
	conc_NaCl=I #Approximate NaCl salinity as sole source of ionic strength

	#########################
	####Initialize variables
	#########################
	#array to hold [H2S], [SO2] concentrations as function of p
	conc_h2s_list_ph2s=np.zeros(len(p_h2s_list))
	conc_so2_list_pso2=np.zeros(len(p_so2_list))

	#array to hold concentrations for buffered system for H2S (Method 1)
	conc_h_list_ph2s=np.zeros(len(p_h2s_list))
	conc_oh_list_ph2s=np.zeros(len(p_h2s_list))
	conc_hs_list_ph2s=np.zeros(len(p_h2s_list))
	conc_s_list_ph2s=np.zeros(len(p_h2s_list))

	#array to hold concentrations for buffered system for SO2 (Method 1)
	conc_h_list_pso2=np.zeros(len(p_so2_list))
	conc_oh_list_pso2=np.zeros(len(p_so2_list))
	conc_hso3_list_pso2=np.zeros(len(p_so2_list))
	conc_so3_list_pso2=np.zeros(len(p_so2_list))
	conc_hs2o5_list_pso2=np.zeros(len(p_so2_list))

	#########################
	####Do calculation
	#########################
	###Henry's Law
	H_h2s=correct_H_NaCl_salinity_jpl(H_0_h2s, h_g_0_h2s, h_t_h2s, T_0_jpl, conc_NaCl) #
	H_so2=correct_H_NaCl_salinity_jpl(H_0_so2, h_g_0_so2, h_t_so2, T_0_jpl, conc_NaCl) #


	conc_h2s_list_ph2s=get_aqueous_concentration(p_h2s_list, H_h2s) #concentration of dissolved H2S from Henry's Law, in M, for specified salinity (T=T_0)
	conc_so2_list_pso2=get_aqueous_concentration(p_so2_list, H_so2) #concentration of dissolved H2S from Henry's Law, in M, for specified salinity (T=T_0)	

	for ind in range(0, len(phi_s_list)):
		#First do H2S
		conc_h2s=conc_h2s_list_ph2s[ind] #specify [H2S] (M)

		conc_h_list_ph2s[ind], conc_oh_list_ph2s[ind], conc_hs_list_ph2s[ind], conc_s_list_ph2s[ind] = solve_buffered_h2s(conc_h2s, pH, I)	
		
		#Second do SO2
		conc_so2=conc_so2_list_pso2[ind] #specify [SO2] (M)
		
		conc_h_list_pso2[ind], conc_oh_list_pso2[ind], conc_hso3_list_pso2[ind], conc_so3_list_pso2[ind], conc_hs2o5_list_pso2[ind] = solve_buffered_so2(conc_so2, pH, I)		

	#########################################################################
	#########################################################################
	#########################################################################
	####MAKE PLOTS: Explore H2S and SO2 together as function of phi_S using Hu et al (2013) as guide
	#########################################################################
	#########################################################################
	#########################################################################

	fig, ax=plt.subplots(3, figsize=(8., 8.), sharex=True, sharey=False)
	markersizeval=5.
	colors=cm.rainbow(np.linspace(0,1,5))

	ax[0].set_title('Speciation of Sulfur-Bearing Molecules\n in Buffered Solution (pH=7)')

	#ax[0].plot(phi_s_list, conc_h2s_list_ph2s, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[0], label=r'H$_{2}$S')
	ax[0].plot(phi_s_list, conc_hs_list_ph2s, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[1], label=r'HS$^{-}$')

	#ax[1].plot(phi_s_list, conc_so2_list_pso2, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[2], label=r'SO$_2$')
	ax[1].plot(phi_s_list, conc_hso3_list_pso2, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[3], label=r'HSO$_{3}$$^{-}$')
	ax[1].plot(phi_s_list, conc_so3_list_pso2, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[4], label=r'SO$_{3}$$^{2-}$')

	ax[2].plot(phi_s_list, p_h2s_list, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[0], label=r'H$_{2}$S')
	ax[2].plot(phi_s_list, p_so2_list, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[2], label=r'SO$_{2}$')


	ax[0].set_xscale('log')
	ax[0].set_ylabel('Concentration (M)')
	ax[0].set_yscale('log')
	ax[0].axvspan(1.*10.**(10.), 1.*10.**(11.5), color='lightgrey', alpha=0.2) #Max S emission, during basaltic plain emplacement, fro Halevy et al 2014 
	#ax[0].axhline(1.e-3, linestyle=':', color='black') #demarcating millimolar concentrations
	ax[0].axhline(1.e-6, linestyle='--', color='black') #demarcating micromolar concentrations
	ax[0].legend(loc='upper left', ncol=1, borderaxespad=0., fontsize=12)

	ax[1].set_xscale('log')
	ax[1].set_ylabel('Concentration (M)')
	ax[1].set_yscale('log')
	ax[1].axvspan(1.*10.**(10.), 1.*10.**(11.5), color='lightgrey', alpha=0.2) #Max S emission, during basaltic plain emplacement, fro Halevy et al 2014 
	ax[1].axhline(1.e-3, linestyle=':', color='black') #demarcating millimolar concentrations
	ax[1].axhline(1.e-6, linestyle='--', color='black') #demarcating micromolar concentrations
	ax[1].legend(loc='upper left', ncol=1, borderaxespad=0., fontsize=12)

	ax[2].set_xlabel(r'$\phi_{S}$ (cm$^{-2}$s$^{-1}$)')
	ax[2].set_xscale('log')
	ax[2].set_xlim([np.min(phi_s_list), np.max(phi_s_list)])
	ax[2].set_ylabel('Surface Pressure (bar)')
	ax[2].set_yscale('log')
	#ax[2].axhline(1.e-3, linestyle=':', color='black') #demarcating millimolar concentrations
	#ax[2].axhline(1.e-6, linestyle='--', color='black') #demarcating micromolar concentrations
	ax[2].axvspan(1.*10.**(10.), 1.*10.**(11.5), color='lightgrey', alpha=0.2) #Max S emission, during basaltic plain emplacement, fro Halevy et al 2014 
	ax[2].legend(loc='upper left', ncol=1, borderaxespad=0., fontsize=12)

	fig.subplots_adjust(wspace=0., hspace=0.1)
	plt.savefig('./Plots/h2s_so2_speciation_phis_'+atmtype+filelabel+'_paper.eps', orientation='portrait',papertype='letter', format='eps')

	print conc_h_list_pso2,'\n', conc_oh_list_ph2s #these better both be 10**-pH...


if run_rt_plot:
	#########################################################################
	#########################################################################
	#########################################################################
	####PLOT RT RESULTS
	#########################################################################
	#########################################################################
	#########################################################################

	########################
	###Plot surface radiance for phi_s's with aerosols and full atm
	########################
	fig6, ax=plt.subplots(1, figsize=(8.5, 6.), sharex=True, sharey=True)
	markersizeval=5.
	colors=cm.rainbow(np.linspace(0,1,len(phi_s_list)))

	elt_list=np.array(['3e9', '1e10', '3e10', '1e11', '3e11', '1e12'])
	label_list=np.array([r'3$\times 10^9$', r'1$\times 10^{10}$', r'3$\times 10^{10}$', r'1$\times 10^{11}$', r'3$\times 10^{11}$', r'1$\times 10^{10}$'])

	for ind in range(0, len(phi_s_list[:-1])):
		elt=elt_list[ind]
		twostr_wav, twostr_toa, twostr_surfint=np.genfromtxt('./RT/TwoStreamOutput/hu2013_'+atmtype+'_sflux'+elt+'_exponential_aerosols.dat', skip_header=1, skip_footer=0,usecols=(2,3,6), unpack=True)
		transmission=twostr_surfint/twostr_toa# 
		
		if ind==0:
			ax.plot(twostr_wav, twostr_toa, markersize=markersizeval, linewidth=3, linestyle='-', color='black', label='TOA')
		
		ax.plot(twostr_wav, twostr_surfint, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[ind], label=r'$\phi_S$='+label_list[ind]+r' cm$^{-2}$ s$^{-1}$')

	ax.axvline(254., linestyle='--', color='black')
	ax.legend(loc=0, ncol=2, borderaxespad=0., fontsize=13)
	ax.set_xlabel('Wavelength (nm)', fontsize=18)
	ax.set_xscale('linear')
	ax.set_xlim([200., 300.])
	ax.set_ylabel(r'Surface Radiance (erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$)', fontsize=18)
	ax.set_yscale('log')
	ax.set_ylim([1.e-2, 1.e4])
	ax.tick_params(axis='y', labelsize=18)	
	ax.tick_params(axis='x', labelsize=18)
	plt.savefig('./Plots/h2s_so2_radiance_twostr_aer_'+atmtype+'.eps', orientation='portrait',papertype='letter', format='eps')
	
	
plt.show()
