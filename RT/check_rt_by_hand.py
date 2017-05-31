# -*- coding: iso-8859-1 -*-
"""
Purpose of this file is to calculate RT using simple 1-layer 2-stream relations from Bohren and others, to validate the remarkable result we are getting that very modest amounts of S8 aerosol are enough to massively suppress surface UV.

We work in a simple case: 255 nm, 1 bar N2 atmosphere
"""
import numpy as np
import pdb

N_atm=2.2e25 #cm**-2, column density of 1 bar N2 atmosphere 
N2_XC=1.20e-25 #XC of N2 at 255 nm
N2_od=N2_XC*N_atm #optical depth of N2

S8_XC_500=1.23# micron**2, XC of S8 aerosol at 500 nm
S8_XC=2.22 #micron**2, XC of s8 aerosol at 255 nm

S8_w0=0.472 #SSA of S8 at 255 nm
S8_g=0.5 #asymmetry parameter of S8 at 255 nm

#######
###User-specified test cases
#######

#phi_S=3e9
#S8_od_500=0.14 #OD of S8 at 500 nm
#observed_suppression=10./33. #Approximate observed suppression. This is what we are trying to match.

##phi_S=1e10
#S8_od_500=0.21 #OD of S8 at 500 nm
#observed_suppression=8./33. #Approximate observed suppression. This is what we are trying to match.

##phi_S=3e10
#S8_od_500=0.45 #OD of S8 at 500 nm
#observed_suppression=4./33. #Approximate observed suppression. This is what we are trying to match.

#phi_S=1e11
S8_od_500=1.03 #OD of S8 at 500 nm
observed_suppression=1.3/33. #Approximate observed suppression. This is what we are trying to match.

##phi_S=3e11
#S8_od_500=2.78 #OD of S8 at 500 nm
#observed_suppression=0.4/33. #Approximate observed suppression. This is what we are trying to match.

#######
###Do calc
#######
S8_od=S8_od_500*S8_XC/S8_XC_500

od_tot=S8_od+N2_od #ignore sulfate aerosols, they have only a small optical depth and all scattering

w0=(1.*N2_od+S8_w0*S8_od)/(od_tot) #N2 is purely scattering
g=(0.*N2_od+S8_g*S8_od)/(od_tot) #N2 is symmetric scattering

K=np.sqrt((1.-w0)*(1.-w0*g))

calc_suppression=np.exp(-K*od_tot)

print observed_suppression
print calc_suppression
print g
pdb.set_trace()