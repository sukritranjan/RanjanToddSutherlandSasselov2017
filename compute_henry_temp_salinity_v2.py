# -*- coding: iso-8859-1 -*-
"""
    Written by: Sukrit Ranjan
    
    Goal of this script is to assess sensitivity of Henry's Law coefficients of H2S and SO2 as a function of temperature and salinity.
    
    The formalisms and constants of Sander (2015) and Burkholder (2015) are compared
    Sander (2015): Compilation of Henry's Law Constants (Version 4) for Water As Solvent
    Burkholder (2015): JPL Publication 15-10, Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation Number 18, Chapter 5.
    
    """

########################
###Import useful libraries
########################
import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.integrate
import scipy.optimize
from matplotlib.pyplot import cm

########################
###Define useful constants, all in CGS (via http://www.astro.wisc.edu/~dolan/constants.html)
########################
bar2atm=0.9869 #1 bar in atm
Pa2bar=1.e-5 #1 Pascal in bar
bar2Pa=1.e5 #1 bar in Pascal

########################
###Specify relevant physical parameters
########################

###
#HENRY'S LAW PARAMETERS FROM SANDER 2015
###
T_0_sander=298.15 #Temperature at which Henry's Law constant was measured at. Taken from standard temperature listed in Table 4 of Sander 2015

H_cp_so2_sander=1.3e-2*1.e-3*bar2Pa#Henry's Law Constant, tabulated for T=298.15K, pure water (0 salinity). Units of M/bar, converted from mol/m^3/Pa by *m^3/(1000L) and *(1bar in Pa)/bar. From Sander et al. 2011 via Sander 2015, column 2 of table 6.
dlnH_cp_d_1overT_so2_sander=2900. #d(ln(H))/(d(1/T))=-delta_{sol}H/R. Units of K. From Sander et al. 2011 via Sander 2015, column 3 of Table 6. See Equations 17 and 19 of Sander 2015 for explanation. Only valid for temperatures near T_0.

H_cp_h2s_sander=1.0e-3*1.e-3*bar2Pa#Henry's Law Constant, tabulated for T=298.15K, pure water (0 salinity). Units of M/bar, converted from mol/m^3/Pa by *m^3/(1000L) and *(1bar in Pa)/bar. From Sander et al. 2011 via Sander 2015, column 2 of table 6.
dlnH_cp_d_1overT_h2s_sander=2100. #d(ln(H))/(d(1/T))=-delta_{sol}H/R. Units of K. From Sander et al. 2011 via Sander 2015, column 3 of Table 6. See Equations 17 and 19 of Sander 2015 for explanation. Only valid for temperatures near T_0.

###
#HENRY'S LAW PARAMETERS FROM BURKHOLDER ET AL 2015
###
T_0_jpl=298.15 #reference temperature in K

#SO2
H_0_so2_jpl=1.36*bar2atm #Henry Law Constant at 298K, from table 5-4 of Burkholder+2015, page 5-156. Units: M/atm converted to M/bar
A_so2_jpl=-39.72
B_so2_jpl=4250.
C_so2_jpl=4.525
h_g_0_so2_jpl=-0.0607 #From table 5-4 of Burkholder+2015, page 5-156. Units: M**-1
h_t_so2_jpl=0.000275 #From table 5-4 of Burkholder+2015, page 5-156. Units: M**-1 K**-1

#H2S
H_0_h2s_jpl=0.102*bar2atm #Henry Law Constant at 298K, from table 5-4 of Burkholder+2015, page 5-156. Units: M/atm converted to M/bar
A_h2s_jpl=-145.2
B_h2s_jpl=8120.
C_h2s_jpl=20.296
h_g_0_h2s_jpl=-0.0333 #From table 5-4 of Burkholder+2015, page 5-156. Units: M**-1
h_t_h2s_jpl=0. #NOT GIVEN FOR H2S. Unknown?

#ion parameters
h_i_na=0.1143 #For Na+. From Table 5-5 of Burkholder+2015, page 5-187
h_i_cl=0.0318 #For Cl-. From Table 5-5 of Burkholder+2015, page 5-187


########################
###Sub Functions: JPL compendium
########################
def correct_H_temp_jpl(A, B, C,T):
	"""
	This function calculates the Henry's Law constant as a function of temperature.
	
	Takes: Coefficients A, B, C from Burkholder+2015 Table 5-4, temperature in K
	Returns: Henry's Law constant, converted from units of M/atm to M/bar
	
	Uses: Equation from Burkholder+2015 pg 5-9.
	"""
	
	H_corrected=np.exp(A+B/T+C*np.log(T)) #from page 5-9 of Burkholder et al 2015
	return H_corrected*bar2atm


###print correct_H_temp_jpl(A_h2s_jpl, B_h2s_jpl, C_h2s_jpl, 273.15)
###print correct_H_temp_jpl(A_h2s_jpl, B_h2s_jpl, C_h2s_jpl, 298.15)
###print correct_H_temp_jpl(A_h2s_jpl, B_h2s_jpl, C_h2s_jpl, 323.15)

###print correct_H_temp_jpl(A_so2_jpl, B_so2_jpl, C_so2_jpl, 273.15)
###print correct_H_temp_jpl(A_so2_jpl, B_so2_jpl, C_so2_jpl, 298.15)
###print correct_H_temp_jpl(A_so2_jpl, B_so2_jpl, C_so2_jpl, 323.15)

def correct_H_NaCl_salinity_jpl(H, h_g_0, h_t, T, conc_NaCl):
	"""
	This function calculates the Henry's Law constant as a function of salinity. It assumes NaCl is the SOLE source of salinity.
	
	Takes: Henry's Law constant, temperature-independent gas-specific Sechenov coefficient, temperature-dependent gas-specific Sechenov coefficient, temperature (K), [NaCl] (M)
	
	Returns: corrected Henry's Law constant
	
	Uses: Sechenov equations, see p 5-187 of Burkholder et al 2015 for example
	"""
	h_g=h_g_0+h_t*(T-T_0_jpl) #all temperature dependence of sechenov constant embedded here, in calculating the gas-specific parameter
	
	K_NaCl=(h_i_na+h_g)*(1.) + (h_i_cl+h_g)*(1.) #I interpret ion index based on the sample calculation of pg 5-187...
	
	####Test case. From Worsnop+1995, K_NaCl for H2S at 278 K is 0.064 M**-1, with values in the literature ranging from 0.741-0.0811. With our numbers, we get K_NaCl for H2S of 0.0795.
	###print K_NaCl
	###pdb.set_trace()
	
	H_corrected=H*10.**(-K_NaCl*conc_NaCl)
	
	return H_corrected

###correct_H_NaCl_salinity_jpl(H_0_h2s_jpl, h_g_0_h2s_jpl, h_t_h2s_jpl, 278., 1.)
	

########################
###Evaluate over range of temperature, salinity
########################

T_list=np.arange(273.15, 324.15, step=1.) #temperatures to try, K
conc_NaCl_list=10.**(np.arange(-6.,1., step=1.)) #[NaCl] to try, M


###
H_so2_jpl_T=correct_H_temp_jpl(A_so2_jpl, B_so2_jpl, C_so2_jpl,T_list) #H_so2 as a function of T, evaluated using JPL function
H_so2_conc_NaCl=correct_H_NaCl_salinity_jpl(H_0_so2_jpl, h_g_0_so2_jpl, h_t_so2_jpl, T_0_jpl, conc_NaCl_list) #H_so2 as a function of [NaCl]

H_h2s_jpl_T=correct_H_temp_jpl(A_h2s_jpl, B_h2s_jpl, C_h2s_jpl,T_list) #H_h2s as a function of T, evaluated using JPL function
H_h2s_conc_NaCl=correct_H_NaCl_salinity_jpl(H_0_h2s_jpl, h_g_0_h2s_jpl, h_t_h2s_jpl, T_0_jpl, conc_NaCl_list) #H_h2s as a function of [NaCl]


##################################
#############Plot: temp
##################################
fig, ax=plt.subplots(2, figsize=(6., 7.), sharex=True, sharey=False)
markersizeval=5.

ax[0].set_title(r'Dependence of H$_{SO2}$ and H$_{H2S}$ on $T$')
ax[0].plot(T_list, H_h2s_jpl_T, marker='s', markersize=markersizeval, linewidth=1, color='black', label='JPL 2015')
ax[0].set_xlabel('T (K)')
ax[0].set_xscale('linear')
ax[0].set_ylabel(r'H$_{H2S}$ (M/bar)')
ax[0].set_yscale('linear')
#ax[0].legend(loc=0, fontsize=10)
ax[0].axvline(T_0_sander, color='blue', linestyle='--')

ax[1].plot(T_list, H_so2_jpl_T, marker='s', markersize=markersizeval, linewidth=1, color='black', label='JPL 2015')
ax[1].set_xlabel('T (K)')
ax[1].set_xscale('linear')
ax[1].set_ylabel(r'H$_{SO2}$ (M/bar)')
ax[1].set_yscale('linear')
#ax[1].legend(loc=0, fontsize=10)
ax[1].axvline(T_0_sander, color='blue', linestyle='--')
ax[1].set_xlim([273.15, 323.15])

plt.savefig('./Plots/h2s_so2_henry_temp_dep.pdf', orientation='portrait',papertype='letter', format='pdf')


##################################
#############Plot: salinity
##################################
fig, ax=plt.subplots(2, figsize=(6., 7.), sharex=False, sharey=False)
markersizeval=5.

ax[0].set_title('Dependence of H$_{SO2}$ and H$_{H2S}$ on [NaCl]')

ax[0].plot(conc_NaCl_list, H_h2s_conc_NaCl, marker='s', markersize=markersizeval, linewidth=1, color='black', label='JPL 2015')
ax[0].set_xlabel('[NaCl] (M)')
ax[0].set_xscale('log')
ax[0].set_ylabel(r'H$_{H2S}$  (M/bar)')
ax[0].set_yscale('log')
ax[0].set_ylim([1.e-2, 1.e0])
#ax[0].legend(loc=0, fontsize=10)
#ax[0].axvline(0.6, color='blue', linestyle='--')

ax[1].plot(conc_NaCl_list, H_so2_conc_NaCl, marker='s', markersize=markersizeval, linewidth=1, color='black', label='JPL 2015')
ax[1].set_xlabel('[NaCl] (M)')
ax[1].set_xscale('log')
ax[1].set_ylabel(r'H$_{SO2}$  (M/bar)')
ax[1].set_yscale('log')
ax[1].set_ylim([1.e-1, 1.e1])
#ax[1].legend(loc=0, fontsize=10)
#ax[1].axvline(0.6, color='blue', linestyle='--')

plt.savefig('./Plots/h2s_so2_henry_nacl_dep.pdf', orientation='portrait',papertype='letter', format='pdf')

plt.show()
