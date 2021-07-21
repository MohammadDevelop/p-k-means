import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import threading
from scipy.spatial.distance import cdist, pdist

#parameters
data_file='dataset.csv'

data = data_file
df = pd.read_csv(data)

df=df.drop(columns=["CUST_ID"])
print(df.head())

df=df.fillna(df.mean())

lables=df.keys()
print(lables)


def simple_k_means(num_cluster):
    global results
    kmeans = KMeans(init="random",n_clusters=num_cluster, n_init=1).fit(df)

    results = results.append({
            "Config": "Random K-means",
            "Clusters": num_cluster,
            "Iterations": kmeans.n_iter_,
            "Inertia": '{:0.2e}'.format(kmeans.inertia_)
    }, ignore_index=True)

    print("Simple K-means - clusters: "+str(num_cluster)+"!")

def k_means_plus_plus(num_cluster):
    global results
    kmeans = KMeans(init="k-means++",n_clusters=num_cluster, n_init=1).fit(df)
    results = results.append({
            "Config": "K-means++",
            "Clusters": num_cluster,
            "Iterations": kmeans.n_iter_,
            "Inertia": '{:0.2e}'.format(kmeans.inertia_)
    }, ignore_index=True)

    print("K-means++ - clusters: " + str(num_cluster) + "!")



results = pd.DataFrame(columns=["Config","Clusters", "Iterations","Inertia"])
for nc in range(2,30,2):
     simple_k_means(nc)

print(results.head())
results.to_csv('result1_simple_kmeans.csv', index=False)

results = pd.DataFrame(columns=["Config","Clusters", "Iterations","Inertia"])
for nc in range(2,30,2):
     k_means_plus_plus(nc)

print(results.head())
results.to_csv('result1_plus_plus.csv', index=False)

