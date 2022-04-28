#!/usr/bin/env python
# coding: utf-8

# In[200]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import scipy.spatial.distance
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# In[140]:


df = pd.read_csv("http://miner2.vsnet.gmu.edu/files/uploaded_files/1649181792_01407_1604554690_4994035_1601384279_9602122_iris_new_data.txt", header=None, sep=" ")


# In[141]:


print(df)


# In[142]:


#Ensure the shape is 150 instances with 4 features.
df.shape


# In[143]:


#X = df.iloc[:, [0, 1, 2, 3]]
X = df.values


# In[144]:


print(X)


# In[146]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(X)
# X = scaler.transform(X)
X= preprocessing.normalize(X)


# In[147]:


print(X)


# In[148]:


np.random.seed(42)


# In[149]:


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# In[213]:


def KMeans(X,K, max_iters):
    
    #Randomly choosing Centroids 
    random_points = np.random.choice(len(X), K, replace=False)
    centroids = [X[point] for point in random_points]
     
    
    #Calculate distance between each data point and available centroids
    distances = distance.cdist(X, centroids ,'euclidean')
        
    #Assign the data point to the closest centroid determined by smallest distance   
    distanceList = []
    for i in distances:
        distanceList.append(np.argmin(i))
    closestPoint = np.array(distanceList)
    
     
    #Step 4
    for _ in range(max_iters): 
        
        #Create an empty list of centroids
        centroids = []
        
        for point in range(K):
            
            #Updating Centroids using mean of data points in cluster
            centroids.append(np.mean([X[x] for x in range(len(X)) if closestPoint[x] == point], axis = 0))
 
        oldCentroids = centroids
        centroids = np.vstack(centroids) #Updated Centroids
        
         
        distances = distance.cdist(X, centroids ,'euclidean')
        #distances = np.sqrt(np.dot(X, X)) - (np.dot(X, centroids) + np.dot(X, centroids)) + np.dot(centroids, centroids)


        newList = []
        for i in distances:
            newList.append(np.argmin(i))
        closestPoint = np.array(newList)
        
        newDistances = []
        for i in range(K):
            dist = np.linalg.norm(oldCentroids[i] - centroids[i])
            newDistances.append(dist)
        
        if sum(newDistances) == 0:
            break

         
    return closestPoint


# In[151]:


labels = KMeans(X,3,100)
print(labels)


# In[152]:


#Ensure cluster labels are only 1 2 3 
for i in range(len(labels)):
    labels[i] = labels[i] + 1


# In[153]:


print(labels)


# In[60]:


#Writing to file
labels.tofile('KPred11.csv', sep = '\n')


# In[61]:


#-------- Part 2 Image Data ---------#


# In[162]:


#Reading from file
df2 = pd.read_csv("http://miner2.vsnet.gmu.edu/files/uploaded_files/1649182019_5350096_1604556007_243332_1601384482_8387134_image_new_test.txt", header=None, sep=",")


# In[163]:


print(df2)


# In[164]:


X2 = df2.values


# In[65]:


print(X2)


# In[165]:


#Standardizing the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X2 = scaler.fit_transform(X2)


# In[166]:


print(X2)


# In[167]:


X2.shape


# In[168]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.gray()
plt.matshow(X2[0].reshape(28,28))


# In[169]:


plt.matshow(X2[1].reshape(28,28))


# In[170]:


#Applying PCA
#from sklearn.decomposition import PCA

#Choose 95% variance
#pca = PCA(n_components = 0.95)
#pca.fit(X2)
#X2_pca = pca.transform(X2)


# In[171]:


#X2_pca.shape


# In[190]:


#Applying T-SNE
from sklearn.manifold import TSNE
 
n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(X2)
tsne_result.shape


# In[191]:


print(tsne_result)


# In[192]:


imageLabels = KMeans(tsne_result, 10, 100)


# In[174]:


print(imageLabels)


# In[175]:


print(imageLabels[:40])


# In[193]:


#Make sure values are from 1-10
for i in range(len(imageLabels)):
    imageLabels[i] = imageLabels[i] + 1


# In[194]:


print(imageLabels[:40])


# In[178]:


imageLabels.tofile('imageK4.csv', sep = '\n')


# In[197]:


#----- Silhouette Score Analysis with varying K -----#


# In[202]:


#K = 2
imageLabels2 = KMeans(tsne_result, 2, 100)
#Calculate Silhouette Score
score2 = silhouette_score(tsne_result, imageLabels2, metric='euclidean')
print(score2)


# In[203]:


#K = 4
imageLabels4 = KMeans(tsne_result, 4, 100)
#Calculate Silhouette Score
score4 = silhouette_score(tsne_result, imageLabels4, metric='euclidean')
print(score4)


# In[204]:


#K = 6
imageLabels6 = KMeans(tsne_result, 6, 100)
#Calculate Silhouette Score
score6 = silhouette_score(tsne_result, imageLabels6, metric='euclidean')
print(score6)


# In[205]:


#K = 8
imageLabels8 = KMeans(tsne_result, 8, 100)
#Calculate Silhouette Score
score8 = silhouette_score(tsne_result, imageLabels8, metric='euclidean')
print(score8)


# In[206]:


#K = 10
imageLabels10 = KMeans(tsne_result, 10, 100)
#Calculate Silhouette Score
score10 = silhouette_score(tsne_result, imageLabels10, metric='euclidean')
print(score10)


# In[207]:


#K = 12
imageLabels12 = KMeans(tsne_result, 12, 100)
#Calculate Silhouette Score
score12 = silhouette_score(tsne_result, imageLabels12, metric='euclidean')
print(score12)


# In[208]:


#K = 14
imageLabels14 = KMeans(tsne_result, 14, 100)
#Calculate Silhouette Score
score14 = silhouette_score(tsne_result, imageLabels14, metric='euclidean')
print(score14)


# In[209]:


#K = 16
imageLabels16 = KMeans(tsne_result, 16, 100)
#Calculate Silhouette Score
score16 = silhouette_score(tsne_result, imageLabels16, metric='euclidean')
print(score16)


# In[210]:


#K = 18
imageLabels18 = KMeans(tsne_result, 18, 100)
#Calculate Silhouette Score
score18 = silhouette_score(tsne_result, imageLabels18, metric='euclidean')
print(score18)


# In[211]:


#K = 20
imageLabels20 = KMeans(tsne_result, 20, 100)
#Calculate Silhouette Score
score20 = silhouette_score(tsne_result, imageLabels20, metric='euclidean')
print(score20)


# In[ ]:




