# -*- coding: iso-8859-1 -*-
"""
The purpose of this script is to generate a file for the indices of refraction for S8 aerosol at UV wavelength.

The script works by taking as input 2 files giving the UV real and imaginary indices of refraction for S8 aerosol, GraphClicked from Tian et al. 2010 (Supplementary Figure S1). We attempted to get in touch with the authors to directly obtain these measurements, but were unable to reach them. Since these data were extracted by GraphClick, I do not trust them past 2 significant figures. 

The script reads these data, interpolates them to a common wavelength scale, plots them to make sure the interpolation has been carried out reasonably, and then prints the results to a file. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy.integrate
import scipy.optimize
from scipy import interpolate as interp

###Read data
wav_i_um, N_i=np.genfromtxt('./Raw_Data/ComplexRefractionIndices/S8_Tian_2010_ImaginaryIndicesOfRefraction.csv', skip_header=0, skip_footer=0, usecols=(0, 1),delimiter=',', unpack=True)#wavelength in microns,  imaginary refractive index.

wav_r_um, N_r=np.genfromtxt('./Raw_Data/ComplexRefractionIndices/S8_Tian_2010_RealIndicesOfRefraction.csv', skip_header=0, skip_footer=0, usecols=(0, 1), delimiter=',', unpack=True)#wavelength in microns,  real refractive index.

###Interpolate
wav_um=np.arange(0.135, 0.515, step=0.001) #wavelength to interpolate to

N_r_func=interp.interp1d(wav_r_um, N_r, kind='linear')
N_i_func=interp.interp1d(wav_i_um, N_i, kind='linear')

N_r_interpolated=N_r_func(wav_um)
N_i_interpolated=N_i_func(wav_um)

###Plot (to ensure match to Figure S1 of Tian+2010, and to make sure interpolation happened OK
fig, (ax1, ax2)=plt.subplots(2, figsize=(8,10), sharex=True)

ax1.set_title('Real Spectral Index of Refraction')
ax1.plot(wav_r_um, N_r, color='black',marker='s', linestyle='None')
ax1.plot(wav_um, N_r_interpolated, color='red',linestyle='-')
ax1.set_xlabel('Wavelength (microns)')
ax1.set_ylabel('Real Index of Refraction')
ax1.set_yscale('linear')
ax1.set_ylim([1.8, 3.2])

ax2.set_title('Imaginary Spectral Index of Refraction')
ax2.plot(wav_i_um, N_i, color='black',marker='s', linestyle='None')
ax2.plot(wav_um, N_i_interpolated, color='red',linestyle='-')
ax2.set_xlabel('Wavelength (microns)')
ax2.set_xscale('linear')
ax2.set_ylim([0.135, 0.405])
ax2.set_ylabel('Imaginary Index of Refraction')
ax2.set_yscale('log')
ax2.set_ylim([1.e-7, 1.e1])

plt.savefig('./Raw_Data/ComplexRefractionIndices/s8_interp.pdf', orientation='portrait',papertype='letter', format='pdf')
plt.show()

###Print
toprint=np.zeros([len(wav_um), 3])
toprint[:,0]=wav_um
toprint[:,1]=N_r_interpolated
toprint[:,2]=N_i_interpolated

np.savetxt('./Raw_Data/ComplexRefractionIndices/S8_Tian_2010_Indices.txt', toprint, delimiter='	', fmt='%1.7e', newline='\n', header='\n'.join(['Formed by reprocess_s8_optical_parameters.py, from digitizing Figure S1 of Tian+2010','wav(microns)	N_r	N_i'])) #REALLY MOLAR CONCENTRATIONS...HOW DOES INTERPRET?
