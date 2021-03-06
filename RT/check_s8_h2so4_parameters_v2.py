# -*- coding: iso-8859-1 -*-
"""
The purpose of this script is to generate plots of the optical parameters for H2SO4 and S8 aersols that we generate, to compare to what is plotted in papers by Tian+2010, Hu+2013, and Kerber+2015, to ensure we are doing things right. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.integrate
import scipy.optimize
from scipy import interpolate as interp
import cPickle as pickle


########################################################################################################################
###First, replicate Tian+2010 S1
########################################################################################################################

###Read data
f=open('./ParticulateOpticalParameters/s8_reff0p01_vareff0p001_lognormal.pickle', 'r')
wav_0p01, sigma_0p01, w_0_0p01, g_0p01, qsca_0p01=pickle.load(f) #units: nm, microns**2, dimless, dimless, dimless

f=open('./ParticulateOpticalParameters/s8_reff0p5_vareff0p001_lognormal.pickle', 'r')
wav_0p5, sigma_0p5, w_0_0p5, g_0p5, qsca_0p5=pickle.load(f) #units: nm, microns**2, dimless, dimless, dimless

f=open('./ParticulateOpticalParameters/s8_reff1_vareff0p001_lognormal.pickle', 'r')
wav_1, sigma_1, w_0_1, g_1, qsca_1=pickle.load(f) #units: nm, microns**2, dimless, dimless, dimless


###Plot
fig, ax1=plt.subplots(1, figsize=(8,6))

ax1.set_title('S8 SSA (Tian+2010 Fig S1 replication)')
ax1.plot(wav_0p01, w_0_0p01, color='black',linestyle='-', label='0.01 um')
ax1.plot(wav_0p5, w_0_0p5, color='black',linestyle=':', label='0.5 um')
ax1.plot(wav_1, w_0_1, color='black',linestyle='--', label='1 um')

ax1.set_xlabel('Wavelength (nm)')
ax1.set_xscale('linear')
ax1.set_xlim([0., 800.])

ax1.set_ylabel('Single Scattering Albedo')
ax1.set_yscale('linear')
ax1.set_ylim([0, 1.2])

ax1.legend(loc=0, borderaxespad=0., fontsize=10) 


plt.savefig('./Plots/tian_2010_s1c_reproduce.pdf', orientation='portrait',papertype='letter', format='pdf')


########################################################################################################################
###Next, replicate Tian+2010 Fig S2. 
########################################################################################################################

###Read data
f=open('./ParticulateOpticalParameters/h2so4_215K_reff0p2_vareff0p001_lognormal.pickle', 'r')
wav, sigma, w_0, g, qsca=pickle.load(f) #units: nm, microns**2, dimless, dimless, dimless

qext=qsca/w_0
###Plot
fig, (ax1, ax2, ax3)=plt.subplots(3, figsize=(8,10), sharex=True)

ax1.set_title('H2SO4 SSA (Tian+2010 Fig S2 replication)')

ax1.plot(wav, qext, color='black',linestyle='-')
ax1.set_ylabel('Q_ext')
ax1.set_yscale('log')
ax1.set_ylim([5.e-2, 5.e0])

ax2.plot(wav, w_0, color='black',linestyle='-')
ax2.set_ylabel('SSA')
ax2.set_yscale('log')
ax2.set_ylim([1.e-4, 2.e0])

ax3.plot(wav, g, color='black',linestyle='-')
ax3.set_ylabel('Asymmetry factor')
ax3.set_yscale('log')
ax3.set_ylim([1.e-4, 1.])

ax3.set_xlabel('Wavelength (nm)')
ax3.set_xscale('log')
ax3.set_xlim([100., 100.*1000.])


plt.savefig('./Plots/tian_2010_s2_reproduce.pdf', orientation='portrait',papertype='letter', format='pdf')


########################################################################################################################
###Next, Hu+2013 Figure 1 (They plot per-molecule cross-section, it should have the same shape as q_ext, q_sca, so we plot those instead. For the 0.1 case. 
########################################################################################################################
r_v=0.0635 # may need to further correct
r_eff=0.103
r_S=0.05

###Read data (our generation)
f=open('./ParticulateOpticalParameters/s8_reff0p103_hu2013.pickle', 'r')
wav_s8, sigma_s8, w_0_s8, g_s8, qsca_s8=pickle.load(f) #units: nm, microns**2, dimless, dimless, dimless
###sigma_s8=np.pi*(r_S)**2*(qsca_s8/w_0_s8)


f=open('./ParticulateOpticalParameters/h2so4_215K_reff0p103_hu2013.pickle', 'r')
wav_h2so4, sigma_h2so4, w_0_h2so4, g_h2so4, qsca_h2so4=pickle.load(f) #units: nm, microns**2, dimless, dimless, dimless
###sigma_h2so4=np.pi*(r_S)**2*(qsca_h2so4/w_0_h2so4)


###Read data (extracted from Hu)
wav_s8_hu_ext, sigma_s8_hu_ext=np.genfromtxt('./Raw_Data/hu_2013_s8_ext.csv', skip_header=0, skip_footer=0, usecols=(0, 1),delimiter=',', unpack=True)#wavelength in microns,  imaginary refractive index.
wav_s8_hu_sca, sigma_s8_hu_sca=np.genfromtxt('./Raw_Data/hu_2013_s8_sca.csv', skip_header=0, skip_footer=0, usecols=(0, 1),delimiter=',', unpack=True)#wavelength in microns,  imaginary refractive index.
wav_h2so4_hu, sigma_h2so4_hu=np.genfromtxt('./Raw_Data/hu_2013_h2so4.csv', skip_header=0, skip_footer=0, usecols=(0, 1),delimiter=',', unpack=True)#wavelength in microns,  imaginary refractive index.


###Plot
fig, (ax1, ax2, ax3)=plt.subplots(3, figsize=(8,10), sharex=True)

ax1.set_title('S8 (Hu+2013 Fig 1 comparison)')
ax1.plot(wav_s8, sigma_s8*4.9e-19/(r_v)**3., color='black',linestyle=':')
ax1.plot(wav_s8, sigma_s8*4.9e-19/(r_v)**3.*w_0_s8, color='orange',linestyle=':')
ax1.plot(wav_s8_hu_ext*1000., sigma_s8_hu_ext, color='black', marker='s')
ax1.plot(wav_s8_hu_sca*1000., sigma_s8_hu_sca, color='orange', marker='s')
ax1.set_ylabel('sigma (cm2/molecule)')
ax1.set_yscale('log')
#ax1.set_ylim([5.e-2, 5.e0])

ax2.set_title('H2SO4 (Hu+2013 Fig 1 comparison)')
ax2.plot(wav_h2so4, sigma_h2so4*2.1e-19/(r_v)**3, color='black',linestyle=':')
ax2.plot(wav_h2so4, sigma_h2so4*2.1e-19/(r_v)**3.*w_0_h2so4, color='orange',linestyle=':')
ax2.plot(wav_h2so4_hu*1000., sigma_h2so4_hu, color='black', marker='s')
ax2.set_ylabel('sigma (cm2/molecule)')
ax2.set_yscale('log')
#ax2.set_ylim([5.e-2, 5.e0])

ax3.set_title('Asymmetry parameter comparison')
ax3.plot(wav_s8, g_s8, color='red',linestyle=':', label='S8')
ax3.plot(wav_h2so4, g_h2so4, color='blue',linestyle=':', label='H2SO4')
ax3.set_ylabel('g')
ax3.set_yscale('linear')
#ax3.set_ylim([5.e-2, 5.e0])

ax3.set_xlabel('Wavelength (nm)')
ax3.set_xscale('log')
ax3.set_xlim([100., 1000.])
ax3.legend(loc=0, borderaxespad=0., fontsize=10) 


plt.savefig('./Plots/hu_2013_f1_partiallyreproduce.pdf', orientation='portrait',papertype='letter', format='pdf')

###Show plots
plt.show()

