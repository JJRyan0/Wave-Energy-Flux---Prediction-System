
# coding: utf-8

# # Renewable Energy: Predicting Wave Energy Flux of oceanographic sensor data from Coastal Wave Bouys with LSTM Recurrent Neural Networks (RNN's) with a time variant
# 
# ### Deep Learning
# 
# Industry: Renewable Energy 
# 
# Question: how do we predict the wave energy flux in KW per unit of wave-crest length? 
# 
# Method: 1
# Recurrent Neural Networks RNN
# 
# This notebook presents the feasibility of implementing of an Artificial Neural Networks (ANN) model designed to recognise patterns in data offering prediction capabilities to extract insight of oceangraphic wave data collected by a number of wave buoys situated at 3 locations off the west coast of Ireland.
# 
# In current times, alternate renewable energy resources are more of a necessity in particular harnessing energy of the ocean, wind and wave energy is being considered as potential solutions to tackle the unset of increased climatic damage. Renewable wave energy has the potential to massively contribute to the production of electricity. However due to high cost to extract and convert this energy to electricity, it is also a *"major challenge to deploy wave energy converters in deep water regions around the coast where maximum energy is mass is produced"* (R.Waters, 2008). 
# 
# A strategy to minimize extraction costs can be to predict the characteristics of the waves, opening up the opportunity of lower operational costs as maximum energy load can be accurately known days or even hours in advance. Analysis of oceanographic wave data calculated in real-time from near shore wave buoy sensors offer an opportunity to better monitor behavior of waves near shore and predict future patterns allowing more frequent and timely energy transmission from shallow water extractors to un-waiting smart electricity grids onshore. 
# 
# Wave monitoring is a challenging aspect; however the emergence of a cognitive Internet of Things (IoT) technology can potentially provide a smart solution, creating a real-time link from ocean wave energy to electricity flowing on the national grid. 
# 
# Machine learning automation using Artificial Neural Networks have the opportunity to thrive on such data often transmitted via MQTT protocols from various sensors at sea to onshore analytical platforms. Futuremore this study shows that accuracy of sensors from monitoring stations, ships and buoys cause restrictions in the performance of existing forecasting models. 
# 
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">**Note:** Citation:  Data Source:Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.. Datasource: http://data.marine.ie/Dataset/Details/20973#</div>
# 
# 
# Created by John Ryan August 2016

# **Required Libraries**

# In[1]:

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from IPython.display import display, Math, Latex


# **1. Data Import**

# In[2]:

wavex = pd.read_csv('C:\\data\\network.csv')#Import data to a Pandas DataFrame
wavex.head(4)


# **2. Data Cleaning and Preperation for Time Series**

#    **2.1.1 Outliers Detection**

# **2. 2 Treating Missing Values**

# In[4]:

#check to see which features are missing the data
wavex.apply(lambda x: sum(x.isnull()), axis=0)


# In[5]:

#Fill in the NaN values with the most common education type in the data
newwave = wavex.fillna(wavex.mean())


# In[6]:

newwave.apply(lambda x: sum(x.isnull()), axis=0)


# **New feature engineering**
# 
# - Create new column converting the significant wave height from cm to meters.

# - Create new target variable from wave energy flux formula below.

# In[7]:

display(Math(r'P = \frac{\rho g^2} {64\pi}H^2_m0 T_e\approx   (0.5 \frac {kW}{m^3 s}) H^2_m0 T_e,'))
display(Math(r'Results = kW / metre  '))


# ##### P = wave energy flux per unit of wave-crest length 
# 
# ##### Hm0 = the significant wave height in Metres
# 
# ##### Te = wave energy period in seconds 
# 
# ##### ρ the water density and 
# 
# ##### g the acceleration by gravity
# 
# ##### Result =  wave power in kilowatts (kW) per metre of wavefront length

# ### Plotting time series with matplotlib

# In[16]:

wavex['time'] = pd.to_datetime(wavex['time'])
indexed_df = wavex.set_index('time')
indexed_df.head(6)


# In[17]:

ts = indexed_df['WaveEnergyFlux']
ts.head(5)


# ### Uni-Variate Analysis

# In[20]:

plt.plot(ts)
plt.title("Wave Energy Flux")
plt.ylabel("wave power in kilowatts (kW) per metre")
plt.grid(True)
plt.show()


# In[26]:

ts_day = ts.resample('D').mean()
ts_day


# In[27]:

plt.plot(ts_day)
plt.title("Wave Energy Flux")
plt.ylabel("wave power in kilowatts (kW) per metre")
plt.grid(True)
plt.show()


# ### Check for stationarity

# In[ ]:




# ### Transformation
# 
# #### Scale the data - Standard Scaler

# In[ ]:




# In[ ]:

newwave.describe()


# **Cross Validate for Ordinal Data**
# 
# split data 60, test 40

# In[ ]:

#Cross - Validation - split the data into 70% training and the remainder for testing the model
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)


# **One hot Encoder**

# In[ ]:

#Label encoder tranforms any label or attribute for input to the algorithim 
#we can also see some missing values in the top few rows of the data set these will also
#need to be treated in a suitable mannor.
for feature in df.columns:
    if df[feature].dtype=='object':
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
df.tail(3)


# In[ ]:




# **Build & Train the Deep Learning Model**

# In[ ]:

from keras.models import sequential
from keras.layers import Dense, Activation

model = Sequential([Dense(40, input_shape=(1,))
    ])
model.fit(xtrain,ytrain, epochs=120, batch_size=50)


# **Compilation**
# 
# 
# 

# In[ ]:

model.compile(optimizer= 'rmsprop', loss='mse')


# **Make Predictions**

# In[ ]:




# In[ ]:




# **Evaluate Model Performance**

# In[ ]:




# In[ ]:




# ###Gradient Descent
