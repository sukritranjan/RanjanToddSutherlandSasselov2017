# -*- coding: iso-8859-1 -*-
"""
This file contains function definitions and calls that create the atmospheric profile files (T, P, and gas molar concentrations as a function of altitude), that form inputs into our radiative transfer codes. There are two broad families of such files:

I) VALIDATION CASES: these include form_profiles_primitive_earth_rugheimer and form_profiles_wuttke. These functions, once called, generate atmospheric profiles files as well as files containing the TOA solar input that we can use to reproduce the calculations of Rugheimer et al (2015) and the measurements of Wuttke et al (2006), as validation cases.

II) RESEARCH CASES: these are the calls that create the feedstock files used in our study. They include form_spectral_feedstock_youngmars, which is used to define a uniform TOA solar flux file for the young Mars. They also include calls to generate_profiles_cold_dry_mars and generate_profiles_volcanic_mars, which are defined in the file mars_atmosphere_models.py, to give the atmospheric profiles for atmospheres with user specified P_0(CO2), T_0, and specified SO2 and H2S loading levels (latter file only)

All file generation calls are at the end of the respective section.
"""
import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy.stats
from scipy import interpolate as interp
import prebioticearth_atmosphere_models as prebioticearth
import cookbook

bar2Ba=1.0e6 #1 bar in Ba
k=1.3806488e-16 #Boltzmann Constant in erg/K


############################################
###RUN
############################################

#Generate T/P profiles and mixing ratios for generic prebiotic atmospheres
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0.9, 0., 0., 'hu2013_co2rich_sflux0_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0.9, 3.e-10, 4.e-10, 'hu2013_co2rich_sflux3e9_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0.9, 9.e-10, 1.e-9, 'hu2013_co2rich_sflux1e10_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0.9, 3.e-9, 9.e-9, 'hu2013_co2rich_sflux3e10_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0.9, 7.e-9, 5.e-8, 'hu2013_co2rich_sflux1e11_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0.9, 1.e-8, 2.e-7, 'hu2013_co2rich_sflux3e11_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0.9, 3.e-8, 7.e-7, 'hu2013_co2rich_sflux1e12_exponential')

prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0., 0., 0., 'hu2013_n2rich_sflux0_exponential') #so2, h2s
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0., 4.e-12, 9.e-14, 'hu2013_n2rich_sflux3e9_exponential') #so2, h2s
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0., 1.e-11, 9.e-14, 'hu2013_n2rich_sflux1e10_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0., 5.e-10, 2.e-13, 'hu2013_n2rich_sflux3e10_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0., 2.e-10, 7.e-13, 'hu2013_n2rich_sflux1e11_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0., 2.e-9, 4.e-10, 'hu2013_n2rich_sflux3e11_exponential')
prebioticearth.generate_profiles_prebioticearth_exponential(1., 288.,0., 2.e-8, 1.e-7, 'hu2013_n2rich_sflux1e12_exponential')



