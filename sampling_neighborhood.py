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

    centroids= pd.DataFrame(columns=lables)

    for i in range(0,num_cluster):
        samples_=samples.sample(frac=0.8,replace=False)
        ary = cdist(samples_, samples_, 'euclid')
        distances = pd.DataFrame(ary)
        neighborhood_threshold = distances.mean().mean() / (num_cluster*num_cluster)
        neighborhood_counts=distances[distances < neighborhood_threshold ].count()
        max_idx=neighborhood_counts.idxmax()
        center_=samples_.iloc[max_idx]
        centroids.append(center_)
        centroids.loc[len(centroids)] = center_
        #n_idxs=distances[distances[max_idx] < neighborhood_threshold].index
        #samples=samples.drop(samples.index[n_idxs])

        print(str(len(centroids))+" Selected Index="+str(max_idx)+" with "+str(neighborhood_counts[max_idx])+" neighbors. starting "+str(len(distances))+" samples")
    return centroids.to_numpy()

def k_means_modified(num_cluster):
    global results
    cent_=compute_centroids(num_cluster,df)
    print(cent_.shape)
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
results.to_csv('result1_modified_sam_nei.csv', index=False)