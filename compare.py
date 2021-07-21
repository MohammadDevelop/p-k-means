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

results = pd.DataFrame(columns=["Config","Clusters", "Iterations","Inertia"])

def simple_k_means(num_cluster):
    global results
    kmeans = KMeans(init="random",n_clusters=num_cluster).fit(df)

    results = results.append({
            "Config": "Random K-means",
            "Clusters": num_cluster,
            "Iterations": kmeans.n_iter_,
            "Inertia": '{:0.2e}'.format(kmeans.inertia_)
    }, ignore_index=True)

    print("Simple K-means - clusters: "+str(num_cluster)+"!")

def k_means_plus_plus(num_cluster):
    global results
    kmeans = KMeans(init="k-means++",n_clusters=num_cluster).fit(df)
    results = results.append({
            "Config": "K-means++",
            "Clusters": num_cluster,
            "Iterations": kmeans.n_iter_,
            "Inertia": '{:0.2e}'.format(kmeans.inertia_)
    }, ignore_index=True)

    print("K-means++ - clusters: " + str(num_cluster) + "!")

def k_means_modified(num_cluster):
    global results
    ary = cdist(df, df, 'euclid')
    distances=pd.DataFrame(ary)
    print(distances.head())
    print(len(distances))
    neighborhood_threshold=distances.mean().mean()/num_cluster

    print(distances[distances < neighborhood_threshold ].count() )
    # for i in range(0,len(distances)):
    #     for j in range(i+1, len(distances)):
    #         if(distances[i][j]<neighborhood_threshold):
    #             #print(i+j)


    kmeans = KMeans(init="k-means++",n_clusters=num_cluster).fit(df)
    results = results.append({
            "Config": "K-means++",
            "Clusters": num_cluster,
            "Iterations": kmeans.n_iter_,
            "Inertia": '{:0.2e}'.format(kmeans.inertia_)
    }, ignore_index=True)

    print("K-means++ - clusters: " + str(num_cluster) + "!")

k_means_modified(10)

# threads = []
#
# for nc in range(2,50):
#     t = threading.Thread(target=simple_k_means,args=(nc,))
#     threads.append(t)
#
# for nc in range(2,50):
#     t = threading.Thread(target=k_means_plus_plus,args=(nc,))
#     threads.append(t)
#
# for x in threads:
#      x.start()
#
# for x in threads:
#      x.join()
#
# print(results.head())
# results.to_csv('result1.csv', index=False)