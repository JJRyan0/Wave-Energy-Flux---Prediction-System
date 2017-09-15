
# coding: utf-8

# In[4]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[5]:

from IPython.display import display, Math, Latex


# In[6]:

display(Math(r'P = \frac{\rho g^2} {64\pi}H^2_m0 T_e\approx   (0.5 \frac {kW}{m^3 s}) H^2_m0 T_e,'))
display(Math(r'Results = kW / metre  '))

P = wave energy flux per unit of wave-crest length 
Hm0 = the significant wave height in Metres
Te = wave energy period in seconds, 
ρ the water density and 
g the acceleration by gravity
Result =  wave power in kilowatts (kW) per metre of wavefront length