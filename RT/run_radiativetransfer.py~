# -*- coding: iso-8859-1 -*-
"""
Purpose of this file is to run the uv_radtrans function from radiativetransfer.py to generate the surface radiance calculations used to derive the results in our paper.

"""

import radiativetransfer_v2 as rt
import numpy as np
import pdb

#########################################################################################################################
###CO2-N2 Atmospheres (Aerosols), Exponential
#########################################################################################################################
elt_list=np.array(['0', '3e9', '1e10', '3e10', '1e11', '3e11', '1e12'])

atmtype='co2rich'
h2so4_mr_list=np.array([0., 1.5e-10, 4.2e-10, 1.4e-9, 3.5e-9, 3.6e-9, 3.6e-9])
s8_mr_list=np.array([0., 0., 0., 1.7e-11, 4.7e-10, 3.2e-9, 8.9e-9])
N_atm=1.45e25 #in cm**-2, calculated from N=P/(g*mu), where mu=42.4 amu
H_atm=5.8e5#in cm, calculated from H=kT/(g*mu), where T=288 K, g,981 cm/s^2, mu=42.4 amu

###atmtype='n2rich'
###h2so4_mr_list=np.array([0., 4.7e-13, 8.5e-13, 1.1e-11, 7.8e-10, 1.6e-10, 1.4e-10])
###s8_mr_list=np.array([0., 2.7e-10, 4.1e-10, 8.8e-10, 2.0e-9, 5.4e-9, 2.0e-8])
###N_atm=2.19e25 #in cm**-2, calculated from N=P/(g*mu), where mu=42.4 amu
###H_atm=8.7e5#in cm, calculated from H=kT/(g*mu), where T=288 K, g,981 cm/s^2, mu=42.4 amu

h2so4_mr_to_od=N_atm*(2.1e-19/(0.0635)**3.)*(4.99e-3) #Converts H2SO4 aersol mixing ratios to optical depths. Uses:Hu+2013 description, with D_S=0.1 micron ---> r_eff=0.103 micron, r_v=0.0635 micron. XC at 500 nm is taken from file.  
s8_mr_to_od=N_atm*(4.9e-19/(0.0635)**3.)*(1.23e-2) #Converts H2SO4 aersol mixing ratios to optical depths. Uses:Hu+2013 description, with D_S=0.1 micron ---> r_eff=0.103 micron, r_v=0.0635 micron. XC at 500 nm is taken from file.  

h2so4_od_list=h2so4_mr_list*h2so4_mr_to_od
s8_od_list=s8_mr_list*s8_mr_to_od

total_od_list=h2so4_od_list+s8_od_list 
print total_od_list #for comparison to bottom part of Figure 5 of Hu+2013, as a check

for ind in range(0, len(elt_list)):
	elt=elt_list[ind]
	h2so4_od=h2so4_od_list[ind]
	s8_od=s8_od_list[ind]

	rt.uv_radtrans(z_upper_limit=64.e5, z_step=1.e5, inputatmofilelabel='hu2013_'+atmtype+'_sflux'+elt+'_exponential', outputfilelabel='aerosols', inputspectrafile='general_youngsun_earth_highres_NUV_spectral_input.dat',TDXC=False, DeltaScaling=True, SZA_deg=48.2, albedoflag='uniformalbedo',uniformalbedo=0.2, includedust=False, includeco2cloud=False,includeh2ocloud=False, includes8aer=True, tau_s8aer=s8_od, s8aerparamsfile='s8_reff0p103_hu2013.pickle', includeh2so4aer=True,tau_h2so4aer=h2so4_od, h2so4aerparamsfile='h2so4_215K_reff0p103_hu2013.pickle', H_atm=H_atm)

#########################################################################################################################
###CO2-N2 Atmospheres (No Aerosols), Exponential
#########################################################################################################################
elt_list=np.array(['0', '3e9', '1e10', '3e10', '1e11', '3e11', '1e12'])

for elt in elt_list:
	rt.uv_radtrans(z_upper_limit=64.e5, z_step=1.e5, inputatmofilelabel='hu2013_co2rich_sflux'+elt+'_exponential', outputfilelabel='no_aerosols', inputspectrafile='general_youngsun_earth_highres_widecoverage_spectral_input.dat',TDXC=False, DeltaScaling=False, SZA_deg=48.2, albedoflag='uniformalbedo',uniformalbedo=0.2, includedust=False, includeco2cloud=False,includeh2ocloud=False)

	###rt.uv_radtrans(z_upper_limit=64.e5, z_step=1.e5, inputatmofilelabel='hu2013_n2rich_sflux'+elt+'_exponential', outputfilelabel='no_aerosols', inputspectrafile='general_youngsun_earth_highres_widecoverage_spectral_input.dat',TDXC=False, DeltaScaling=False, SZA_deg=48.2, albedoflag='uniformalbedo',uniformalbedo=0.2, includedust=False, includeco2cloud=False,includeh2ocloud=False)


###########rt.uv_radtrans(z_upper_limit=64.e5, z_step=1.e5, inputatmofilelabel='hu2013_co2rich_sflux3e11_dryadiabat_relH=1', outputfilelabel='no_aerosols', inputspectrafile='general_youngsun_earth_highres_widecoverage_spectral_input.dat',TDXC=False, DeltaScaling=False, SZA_deg=48.2, albedoflag='uniformalbedo',uniformalbedo=0.2, includedust=False, includeco2cloud=False,includeh2ocloud=False)


