#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import time
import math
from sklearn.neighbors import DistanceMetric
from tqdm import tqdm_notebook as tqdm
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import spatial


# import datasets from files
def importdataset(filename):
    df = pd.read_csv(filename, header=None, sep=" ")
    df.columns = [i for i in range(301)];
    print(df.shape)
    return df



# files name
filename_animal    = './data/animals'
filename_countries = './data/countries'
filename_fruits    = './data/fruits'
filename_veggies   = './data/veggies'


# In[5]:


data_animal        = importdataset(filename_animal)
data_countries     = importdataset(filename_countries)
data_fruits        = importdataset(filename_fruits)
data_veggies       = importdataset(filename_veggies)


# In[6]:


target = [0]*50 + [1] *161 + [2] * 58 + [3] * 60
len(target)


# In[7]:


final_data = pd.concat([data_animal, data_countries, data_fruits, data_veggies])


# In[8]:


classes = final_data[0].tolist()


# In[9]:


# Dropping the name column
final_data = final_data.drop(0,axis=1)


# In[10]:


# Standarizing the data
#final_data = (final_data - final_data.mean())/final_data.std()
final_data.shape


# In[11]:


#final_data.insert(final_data.shape[1], "target", target, True)


# In[12]:


final_data.head()


# In[49]:


def nCr(n,r):
    if n==1:
        return 0
    return (n * (n-1))/2


# In[50]:


def Distance(j, k):
    return math.sqrt(np.sum(np.square(j-k)))  


# In[51]:


def ManhattanDistance(j, k):
    a = j - k
    a =  np.where(a>0,a, -1*a)
    return np.sum(a)
# ManhattanDistance(np.array([ -1, 1, 3, 2 ]),np.array([ 5, 6, 5, 3 ]))


# In[103]:


def CosineDistance(j, k):
    return spatial.distance.cosine(j,k)


# In[115]:


def kmeans(k, n, X, distance):
    '''
    k: clusters
    n: no of iterations
    X: data
    distance = 0 "Euclidean"
    distance = 1 "Manhattan Distance"
    distance = 2 "Cosine Distance"
    '''
    randominx = np.random.permutation(X.shape[0])
    r = randominx[:k]
    centroids = X.iloc[r].values
#     print('Intial Clusters: ', centroids )
    r = [i for i in range(len(centroids))]
    sse = {}
    clusters_items = {}
    
    for i in range(n):
        clusters_items = {}
        #intializing the dictionary
        for j in r:
            clusters_items[j] = []
        # items
        for j in range(X.shape[0]):
            dmin = math.inf
            ditm = -1
            # centroids
            for p in r:
                if distance == 0:
                    dist = Distance(X.iloc[j], centroids[p])
                elif distance == 1:
                    dist = ManhattanDistance(X.iloc[j], centroids[p])
                elif distance == 2:
                    dist = CosineDistance(X.iloc[j], centroids[p])
                if dist < dmin:
                    dmin = dist
                    ditm = p
            clusters_items[ditm].append(j)
        # calculate mean to compute new centroids
        # for each centroids
        temp_c = {}
        sum =0 
        for j in r:
            temp_c[j] = np.mean(X.iloc[clusters_items[j]], axis=0)
            # clusters_items[j] = temp_c[j]
            for l in clusters_items[j]:
                sum = sum + np.sum(np.square(X.iloc[l] - temp_c[j]))
            centroids[j] = temp_c[j]
        sse[i] = sum
        #print("Centroids", centroids)
        r = [i for i in range(len(centroids))]

    return clusters_items, centroids


# In[109]:


def question(data,title,distance):
    '''
    distance = 0 "Euclidean"
    distance = 1 "Manhattan Distance"
    distance = 2 "Cosine Distance"
    '''
    recall_list = []
    precision_list = []
    f1_list = []
    for k in tqdm(range(1, 11)):
        clusters, centroids = kmeans(k, 10, data[:], distance)
        cluster_class_found = {}
        TP_FP=0
        TP=0
        classes = 4
        cluster_table = []
        for i in range(k):
            cluster_table.append([0,0,0,0])
        cluster_table =  cluster_table[:]
        for i in clusters:
            temp = nCr(len(clusters[i]),2)
            cluster_class_found[i] = [target[j] for j in clusters[i]]
            unique, counts = np.unique(cluster_class_found[i], return_counts=True)
            for j in range(len(unique)):
                cluster_table[i][unique[j]] = cluster_table[i][unique[j]] + counts[j]
                if counts[j] > 1:
                    temp_tp = nCr(counts[j], 2)
                    TP = TP + temp_tp
            TP_FP=TP_FP+temp
        Precision = TP/TP_FP
#         print("Precision: ", Precision)
        cluster_FN_sum = []
        cluster_table = np.array(cluster_table)
        cluster_table = np.transpose(cluster_table).tolist()
        for i in range(classes):
            row_sum =0
            for j in range(k-1):
                ele = cluster_table[i][j]
                row_sum = row_sum + ele * sum(cluster_table[i][j+1:])  
            cluster_FN_sum.append(row_sum) 
#         print("Cluster FN Sum: ", cluster_FN_sum)
        FN = sum(cluster_FN_sum)
        Recall = TP/ (TP + FN)
#         print("Recall", Recall)
        F1_Score = 2 * ( (Precision * Recall) / (Precision + Recall) )
        precision_list.append(Precision)
        recall_list.append(Recall)
        f1_list.append(F1_Score)
    
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.plot(range(1,11), f1_list)
    plt.plot(range(1,11), precision_list)
    plt.plot(range(1,11), recall_list)
    plt.legend(["F1-Score", "Precision", "Recall"], loc="upper right")
    plt.xlabel('No Of Clusters: k')
    plt.show()
    return f1_list, precision_list, recall_list


# In[129]:


def q1(k, n, X, distance):
    
    '''
    k: clusters
    n: no of iterations
    X: data
    distance = 0 "Euclidean"
    distance = 1 "Manhattan Distance"
    distance = 2 "Cosine Distance"
    '''
    randominx = np.random.permutation(X.shape[0])
    r = randominx[:k]
    centroids = X.iloc[r].values
#     print('Intial Clusters: ', centroids )
    r = [i for i in range(len(centroids))]
    sse = {}
    clusters_items = {}
    
    for i in range(n):
        clusters_items = {}
        #intializing the dictionary
        for j in r:
            clusters_items[j] = []
        # items
        for j in range(X.shape[0]):
            dmin = math.inf
            ditm = -1
            # centroids
            for p in r:
                if distance == 0:
                    dist = Distance(X.iloc[j], centroids[p])
                elif distance == 1:
                    dist = ManhattanDistance(X.iloc[j], centroids[p])
                elif distance == 2:
                    dist = CosineDistance(X.iloc[j], centroids[p])
                if dist < dmin:
                    dmin = dist
                    ditm = p
            clusters_items[ditm].append(j)
        # calculate mean to compute new centroids
        # for each centroids
        temp_c = {}
        sum =0 
        for j in r:
            temp_c[j] = np.mean(X.iloc[clusters_items[j]], axis=0)
            # clusters_items[j] = temp_c[j]
            for l in clusters_items[j]:
                sum = sum + np.sum(np.square(X.iloc[l] - temp_c[j]))
            centroids[j] = temp_c[j]
        sse[i] = sum
        #print("Centroids", centroids)
        r = [i for i in range(len(centroids))]
    plt.figure(figsize=(8,8))
    plt.title("K-Means Convergence")
    plt.plot(range(n), [sse[i] for i in sse])
    plt.legend(["F1-Score"], loc="upper right")
    plt.xlabel('No Of Interations: k')
    plt.show()
q1(4, 20, final_data[:], 0)


# In[124]:


def q2():
    f, p, r = question(final_data[:],"Normal Euclidean Plot",distance=0)
q2()


# In[125]:


def q3():
    X = final_data[:]
    val = preprocessing.normalize(X)
    X = pd.DataFrame(val)
    f, p, r = question(X,"L2 normalize data",distance=0)
q3()


# In[126]:


def q4():
    f, p, r = question(final_data[:],"Manhattan Distance Plot", distance=1)
q4()


# In[127]:


def q5():
    f, p, r = question(final_data[:],"Cosine Distance Plot", distance=2)
q5()

