#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.datasets import load_breast_cancer


# In[5]:


cancer=load_breast_cancer()


# In[6]:


cancer.keys()


# In[9]:


print(cancer["DESCR"])


# In[11]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[12]:


df.head()


# In[13]:


from sklearn.preprocessing import StandardScaler


# In[14]:


scaler=StandardScaler()


# In[15]:


scaler.fit(df)


# In[17]:


scaled_data=scaler.transform(df)


# In[18]:


from sklearn.decomposition import PCA


# In[19]:


pca=PCA(n_components=2)


# In[20]:


pca.fit(scaled_data)


# In[21]:


x_pca=pca.transform(scaled_data)


# In[22]:


scaled_data.shape


# In[23]:


x_pca.shape


# In[24]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel("First principal component")
plt.ylabel("Second principal component")


# In[ ]:




