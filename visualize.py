# p-k-means Project
# by Mohammad Mahmoodi Varnamkhasti

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("result1_simple_kmeans.csv")
# line 1 for simple kmeans
x1 = df["Clusters"].to_numpy()
y1 = df["Iterations"].to_numpy()
# plotting the line 1 points
plt.plot(x1, y1, label = "simple kmeans")

df = pd.read_csv("result1_modified_sam_nei.csv")
# line 2 for kmeans with sampling neighborhood
x2 = df["Clusters"].to_numpy()
y2 = df["Iterations"].to_numpy()
# plotting the line 2 points
plt.plot(x2, y2, label = "sampling neighborhood")

df = pd.read_csv("result1_plus_plus.csv")
# line 3 for kmeans++
x3 = df["Clusters"].to_numpy()
y3 = df["Iterations"].to_numpy()
# plotting the line 2 points
plt.plot(x3, y3, label = "kmeans++")

plt.xlabel('Clusters')
# Set the y axis label of the current axis.
plt.ylabel('Iterations')
# Set a title of the current axes.
plt.title('Comparing Iterations ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()