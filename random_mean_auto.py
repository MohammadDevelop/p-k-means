# p-k-means Project
# by Mohammad Mahmoodi Varnamkhasti

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import threading
from scipy.spatial.distance import cdist, pdist

#Prepare Data
data_file='dataset.csv'
data = data_file
df = pd.read_csv(data)

df=df.drop(columns=["CUST_ID"])
print(df.head())
df=df.fillna(df.mean())
lables=df.keys()
print(lables)

#Function to Find initial Centroids
def compute_centroids(num_cluster,samples):

    centroids= np.empty((0,17), float)

    mean_fraction=min(2/num_cluster,1.0)
    while len(centroids)<num_cluster:
        samples_=samples.sample(frac=mean_fraction,replace=False)
        mean_vector = samples_.mean().to_numpy()

        centroids=np.vstack([centroids,mean_vector])
        print(str(len(centroids))+"Centroids generated!")
    return centroids

def k_means_modified(num_cluster):
    global results
    cent_=compute_centroids(num_cluster,df)
    kmeans = KMeans(init=cent_,n_clusters=num_cluster,n_init=1).fit(df)
    results = results.append({
            "Config": "modified",
            "Clusters": num_cluster,
            "Iterations": kmeans.n_iter_,
            "Inertia": '{:0.2e}'.format(kmeans.inertia_)
    }, ignore_index=True)

    print("modified - clusters: " + str(num_cluster) + "!")


results = pd.DataFrame(columns=["Config","Clusters", "Iterations","Inertia"])
for nc in range(2,50,1):
     k_means_modified(nc)

print(results.head())
results.to_csv('random_mean_auto_m2.csv', index=False)