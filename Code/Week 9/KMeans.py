#!/usr/bin/env python
# coding: utf-8

# # 0.) Import and Clean data

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[5]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[6]:


df = pd.read_csv("Country-data.csv", sep = ",")


# In[7]:


df.head()


# In[8]:


df.columns


# In[9]:


names = df[["country"]]
X = df.drop(["country"], axis = 1)


# In[10]:


scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)


# # 1.) Fit a kmeans Model with any Number of Clusters

# In[11]:


kmeans = KMeans(n_clusters= 3
                , random_state=42).fit(X_scaled)


# # 2.) Pick two features to visualize across

# In[12]:


X.columns


# In[13]:


# CHANGE THESE BASED ON WHICH IS INTERESTING TO YOU
x1_index = 0
x2_index = 3


plt.scatter(X_scaled[:, x1_index], X_scaled[:, x2_index], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, x1_index], kmeans.cluster_centers_[:, x2_index], marker='o', color='black', s=100)

plt.xlabel(X.columns[x1_index])
plt.ylabel(X.columns[x2_index])
plt.title('Scatter Plot of Customers')
plt.legend(["Group 1", "Center", "Group 2"])
plt.grid()
plt.show()


# # 3.) Check a range of k-clusters and visualize to find the elbow. Test 30 different random starting places for the centroid means
#  

# In[14]:


WCSSs = []
Ks = range(1, 15)
for k in Ks:
    kmeans = KMeans(n_clusters=k, n_init=30, init="random")
    kmeans.fit(X_scaled)
    WCSSs.append(kmeans.inertia_)


# # 4.) Use the above work and economic critical thinking to choose a number of clusters. Explain why you chose the number of clusters and fit a model accordingly.

# In[15]:


plt.plot(WCSSs)
plt.xlabel("n_cluster")
plt.ylabel("WCSSs")
plt.show()


# # 5.) Create a list of the countries that are in each cluster. Write interesting things you notice. Hint : Use .predict(method)

# With the help of elbow criteria, we choose $n\_cluster=2$, because it looks most like a pivot point. As n increase, WCSSs doesn't decrease significantly.

# In[16]:


k = 2
kmeans = KMeans(n_clusters=k, n_init=30, init="random").fit(X_scaled)


# In[17]:


kmeans


# In[18]:


preds = pd.DataFrame(kmeans.predict(X_scaled))
output = pd.concat([preds, X, names], axis=1)


# In[19]:


output


# In[20]:


print("Cluster1:")
list(output[output[0]==0]["country"])


# In[21]:


print("Cluster2:")
list(output[output[0]==1]["country"])


# # 6.) Create a table of Descriptive Statistics. Rows being the Cluster number and columns being all the features. Values being the mean of the centroid. Use the nonscaled X values for interprotation

# In[22]:


Q6DF = pd.concat([preds, X], axis = 1)
Q6DF.groupby(0).mean()


# In[23]:


Q6DF.head()


# # Q7.) Write an observation about the descriptive statistics.

# Rich country with higher income and gdp's people tends to have less children and lower inflation.

# In[24]:


plt.scatter(Q6DF[Q6DF[0]==0].income, Q6DF[Q6DF[0]==0].child_mort)
plt.scatter(Q6DF[Q6DF[0]==1].income, Q6DF[Q6DF[0]==1].child_mort)
plt.xlabel("income")
plt.ylabel("child")
plt.title('Scatter Plot of Customers')
plt.legend(["High Income", "Low Income"])
plt.grid()
plt.show()


# In[25]:


import seaborn as sns
plt.scatter(Q6DF[Q6DF[0]==0].income, Q6DF[Q6DF[0]==0].inflation)
plt.scatter(Q6DF[Q6DF[0]==1].income, Q6DF[Q6DF[0]==1].inflation)
plt.xlabel("income")
plt.ylabel("inflation")
plt.ylim((0, 40))
plt.title('Scatter Plot of Customers')
plt.legend(["High Income", "Low Income"])
plt.grid()
plt.show()

