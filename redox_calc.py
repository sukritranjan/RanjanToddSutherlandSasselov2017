"""
Written by: Sukrit Ranjan

Purpose of this script is to calculate the speciation of S-anions under assumption of dynamic equilibrium, with source atmospheric deposition from the model of Hu+2013, and sink the redox reactions

(A) 2HS- + 4HSO3- -------------------> 3S2O3-- + 3H2O (Siu+1999)
(B) 4SO3-- + H+ ---------------------> 2SO4-- + S2O3-- + H2O

We calculate dissociation using 
1) H2S -------------> HS(-) + H(+)   pKa_1=7.05    1st dissociation     CRC Handbook, 90th Ed, p 8-40, Dissoc. Const. of Inorganics
(2) HS(-) -----------> S(2-) + H(+)   pKa_2=19      2nd dissociation     CRC Handbook, 90th Ed, p 8-40, Dissociation Constants of Inorganics
(3) SO2 + H2O -------> H(+) + HSO3(-) pKa_1=1.86         Bisulfite production       (Neta and Huie 1985)
(4) HSO3(-) ---------> H(+) + SO3(2-) pKa_2=7.2          Sulfate production         (Neta and Huie 1985)
(5) HSO3(-) + SO2 ---> HS2O5(-)       pKa_2=1.5          Disulfite production       (Neta and Huie 1985)

Our calculation requires us to set the lake depth, temperature, and lake catchment to surface area ratio. It is very sensitive to all these parameters, but especially temperature.
"""
#####################################
###Control Switches
#####################################
T=288. #K; temperature
d_col=1.e2 #cm; depth of water column
A_ratio=1. #Ratio between catchment area and lake area; \geq 1.
pH=7. #pH of solution. Assumed to be fixed (buffered)




########################################################################
########################################################################
########################################################################
###SETUP & INPUTS
########################################################################
########################################################################
########################################################################



#####################################
###Import useful libraries
#####################################
import numpy as np
import pdb
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
#####################################
###Set physical constants
#####################################
R=8.314e-3 #kJ mol**-1 K**-1; gas constant. NOTE: This is in SI unlike everything else which is in CGS, be careful!!!

#Unit conversions
bar2atm=0.9869 #1 bar in atm
M2cgs=6.02e23*1.e-3 #convert 1 M (mol/L) to cm^-3
cgs2M=1.e3/(6.02e23) #convert 1 cm^-3 to M (mol/L)

#####################################
###Set value of key parameters (Hu+2013, Halevy+2013, Siu+1999)
#####################################
n_atm=2.4e19 #cm**-3; Density of atm at surface; calculated from Hu+2013 and p=nkT

v_dep_h2s=0.015 #cm/s; deposition velocity of H2S; from Hu+2013
v_dep_so2=1. #cm/s; deposition velocity of SO2; from Hu+2013

k_siu=3.8e3 #M**-2 s**-1; rate coefficient for redox rxn from Siu+1999; T=293K
k_halevy_slowdisp=np.exp(-50./(R*T)) # s**-1; rate coefficient for redox rxn from Halevy+2013, slow disproportionation (Ea=50 kJ/mol)
k_halevy_fastdisp=np.exp(-40./(R*T)) # s**-1; rate coefficient for redox rxn from Halevy+2013, slow disproportionation (Ea=40 kJ/mol)


########################
###Atmospheric mixing ratios (Hu+2013, Figure 5, CO2-rich case)
#############################
#####phi_s_list=np.array([3.e9, 1.e10, 3.e10, 1.e11, 3.e11, 1.e12, 3.e12, 1.e13]) #units: cm**-2 s**-1
#####phi_s_list_labels=np.array(['3.e9', '1.e10', '3.e10', '1.e11', '3.e11', '1.e12', '3.e12', '1.e13']) #for legend

#####mr_h2s_phis=np.array([4.e-10, 1.e-9, 9.e-9, 5.e-8, 2.e-7, 7.e-7, 2.e-6, 9.e-6]) #column-integrated mixing ratios
#####mr_so2_phis=np.array([3.e-10, 9.e-10, 3.e-9, 7.e-9, 1.e-8, 3.e-8, 8.e-8, 3.e-7]) #column-integrated mixing ratios

########################
phi_s_list=np.array([3.e9, 1.e10, 3.e10, 1.e11, 3.e11, 1.e12]) #units: cm**-2 s**-1
phi_s_list_labels=np.array(['3.e9', '1.e10', '3.e10', '1.e11', '3.e11', '1.e12']) #for legend

mr_h2s_phis=np.array([4.e-10, 1.e-9, 9.e-9, 5.e-8, 2.e-7, 7.e-7]) #column-integrated mixing ratios
mr_so2_phis=np.array([3.e-10, 9.e-10, 3.e-9, 7.e-9, 1.e-8, 3.e-8]) #column-integrated mixing ratios

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



########################################################################
########################################################################
########################################################################
###Do calculation
########################################################################
########################################################################
########################################################################

#####################################
###Initialize relevant variables
#####################################

#These will hold the values for the slow disp.
conc_so3_slowdisp_phis=np.zeros(np.shape(phi_s_list))
conc_hso3_slowdisp_phis=np.zeros(np.shape(phi_s_list))

conc_hs_slowdisp_phis=np.zeros(np.shape(phi_s_list))

#These will hold the values for the fast disp.
conc_so3_fastdisp_phis=np.zeros(np.shape(phi_s_list))
conc_hso3_fastdisp_phis=np.zeros(np.shape(phi_s_list))

conc_hs_fastdisp_phis=np.zeros(np.shape(phi_s_list))

####We don't worry about tracking the other species because they are not relevant to the science

#####################################
###Initialize loop
#####################################

for ind in range(0, len(phi_s_list)):
	r_h2s=mr_h2s_phis[ind]
	r_so2=mr_so2_phis[ind]


	#####################################
	###Calculate total S(II-) and S(IV) supply
	#####################################

	s_iim_source=r_h2s*n_atm*v_dep_h2s*A_ratio # cm**-2 s**-1; total flux of H2S into solution.
	s_iv_source=r_so2*n_atm*v_dep_so2*A_ratio # cm**-2 s**-1; total flux of SO2 into solution. 

	#####################################
	###Calculate [S(IV)]
	#####################################
	conc_siv_slowdisp=((s_iv_source-2.*s_iim_source)*cgs2M*d_col**-1)/(k_halevy_slowdisp)
	conc_siv_fastdisp=((s_iv_source-2.*s_iim_source)*cgs2M*d_col**-1)/(k_halevy_fastdisp)

	#####################################
	###Calculate S(IV) speciation
	#####################################	
	conc_so2_slowdisp=conc_siv_slowdisp/(1. + 10.**(pH-pKa_so2_1) + 10.**(pH-pKa_so2_1)*10.**(pH-pKa_so2_2)) #IGNORE HS2O5 for now b/c trace and would make it quadratic.
	conc_hso3_slowdisp=conc_so2_slowdisp*10.**(pH-pKa_so2_1)
	conc_so3_slowdisp=conc_so2_slowdisp*10.**(pH-pKa_so2_1)*10.**(pH-pKa_so2_2)
	
	conc_so2_fastdisp=conc_siv_fastdisp/(1. + 10.**(pH-pKa_so2_1) + 10.**(pH-pKa_so2_1)*10.**(pH-pKa_so2_2)) #IGNORE HS2O5 for now b/c trace and would make it quadratic.
	conc_hso3_fastdisp=conc_so2_fastdisp*10.**(pH-pKa_so2_1)
	conc_so3_fastdisp=conc_so2_fastdisp*10.**(pH-pKa_so2_1)*10.**(pH-pKa_so2_2)
	
	#####################################
	###Calculate [S(II-)]
	#####################################
	conc_hs_slowdisp=(s_iim_source*cgs2M*d_col**-1)/(0.66*k_siu*conc_hso3_slowdisp**2.)
	conc_hs_fastdisp=(s_iim_source*cgs2M*d_col**-1)/(0.66*k_siu*conc_hso3_fastdisp**2.)
	
	#####################################
	###Load into external variables
	#####################################
	conc_hso3_slowdisp_phis[ind]=conc_hso3_slowdisp
	conc_so3_slowdisp_phis[ind]=conc_so3_slowdisp
	conc_hs_slowdisp_phis[ind]=conc_hs_slowdisp
	
	conc_hso3_fastdisp_phis[ind]=conc_hso3_fastdisp
	conc_so3_fastdisp_phis[ind]=conc_so3_fastdisp
	conc_hs_fastdisp_phis[ind]=conc_hs_fastdisp

#########################################################################
#########################################################################
#########################################################################
####MAKE PLOTS: Plot S-anions
#########################################################################
#########################################################################
#########################################################################
fig, ax=plt.subplots(2, figsize=(8., 8.), sharex=True, sharey=False)
markersizeval=5.
colors=cm.rainbow(np.linspace(0,1,3))

ax[0].set_title('S-Anion Concentrations using Dynamic Equilibrium\n (Buffered Solution, pH=7)', fontsize=14)

ax[0].plot(phi_s_list, conc_hs_slowdisp_phis, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[0], label=r'HS$^{-}$ (Slow Disprop.)')
ax[0].plot(phi_s_list, conc_hs_fastdisp_phis, marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[0], label=r'HS$^{-}$ (Fast Disprop.)')

ax[1].plot(phi_s_list, conc_hso3_slowdisp_phis, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[1], label=r'HSO$_{3}$$^{-}$ (Slow Disprop.)')
ax[1].plot(phi_s_list, conc_hso3_fastdisp_phis, marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[1], label=r'HSO$_{3}$$^{-}$ (Fast Disprop.)')

ax[1].plot(phi_s_list, conc_so3_slowdisp_phis, marker='s', markersize=markersizeval, linewidth=1, linestyle='-', color=colors[2], label=r'SO$_{3}$$^{2-}$ (Slow Disprop.)')
ax[1].plot(phi_s_list, conc_so3_fastdisp_phis, marker='s', markersize=markersizeval, linewidth=1, linestyle='--', color=colors[2], label=r'SO$_{3}$$^{2-}$ (Fast Disprop.)')


ax[0].set_xscale('log')
ax[0].set_ylabel('Concentration (M)', fontsize=14)
ax[0].set_yscale('log')
ax[0].axvspan(1.*10.**(10.), 1.*10.**(11.5), color='lightgrey', alpha=0.2) #Max S emission, during basaltic plain emplacement, fro Halevy et al 2014 
ax[0].axhline(1.e-3, linestyle=':', color='black') #demarcating millimolar concentrations
ax[0].axhline(1.e-6, linestyle='--', color='black') #demarcating micromolar concentrations
ax[0].legend(loc='upper left', ncol=1, borderaxespad=0., fontsize=10)

ax[1].set_xscale('log')
ax[1].set_ylabel('Concentration (M)', fontsize=14)
ax[1].set_yscale('log')
ax[1].axvspan(1.*10.**(10.), 1.*10.**(11.5), color='lightgrey', alpha=0.2) #Max S emission, during basaltic plain emplacement, fro Halevy et al 2014 
ax[1].axhline(1.e-3, linestyle=':', color='black') #demarcating millimolar concentrations
ax[1].axhline(1.e-6, linestyle='--', color='black') #demarcating micromolar concentrations
ax[1].legend(loc='best', ncol=1, borderaxespad=0., fontsize=14)

ax[1].set_xlabel(r'$\phi_{S}$ (cm$^{-2}$s$^{-1}$)', fontsize=14)
ax[1].set_xscale('log')
ax[1].set_xlim([np.min(phi_s_list), np.max(phi_s_list)])

ax[1].axvspan(1.*10.**(10.), 1.*10.**(11.5), color='lightgrey', alpha=0.2) #Max S emission, during basaltic plain emplacement, fro Halevy et al 2014 
ax[1].legend(loc='best', ncol=1, borderaxespad=0., fontsize=10)

ax[0].tick_params(axis='y', labelsize=12)	
ax[0].tick_params(axis='x', labelsize=12)
ax[1].tick_params(axis='y', labelsize=12)	
ax[1].tick_params(axis='x', labelsize=12)
ax[1].tick_params(axis='y', labelsize=12)	
ax[1].tick_params(axis='x', labelsize=12)

fig.subplots_adjust(wspace=0., hspace=0.1)
plt.savefig('./Plots/redox_paper.eps', orientation='portrait',papertype='letter', format='eps')


#pdb.set_trace()
plt.show()
