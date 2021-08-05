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

# df = pd.read_csv("result1_modified_sam_nei.csv")
# # line 2 for kmeans with sampling neighborhood
# x2 = df["Clusters"].to_numpy()
# y2 = df["Iterations"].to_numpy()
# # plotting the line 2 points
# plt.plot(x2, y2, label = "sampling neighborhood")

df = pd.read_csv("result1_plus_plus.csv")
# line 3 for kmeans++
x3 = df["Clusters"].to_numpy()
y3 = df["Iterations"].to_numpy()
# plotting the line 2 points
plt.plot(x3, y3, label = "kmeans++")


# df = pd.read_csv("random_mean10.csv")
# # line 7 for random mean 10
# x7 = df["Clusters"].to_numpy()
# y7 = df["Iterations"].to_numpy()
# # plotting the line 2 points
# plt.plot(x7, y7, label = "random mean 10")

# df = pd.read_csv("random_mean25.csv")
# # line 4 for random mean 25
# x4 = df["Clusters"].to_numpy()
# y4 = df["Iterations"].to_numpy()
# # plotting the line 2 points
# plt.plot(x4, y4, label = "random mean 25")


# df = pd.read_csv("random_mean50.csv")
# # line 5 for random mean 50
# x5 = df["Clusters"].to_numpy()
# y5 = df["Iterations"].to_numpy()
# # plotting the line 2 points
# plt.plot(x5, y5, label = "random mean 50")



# df = pd.read_csv("random_mean75.csv")
# # line 5 for random mean 75
# x6 = df["Clusters"].to_numpy()
# y6 = df["Iterations"].to_numpy()
# # plotting the line 2 points
# plt.plot(x6, y6, label = "random mean 75")

df = pd.read_csv("random_mean_auto.csv")
# line 9 for random mean Auto
x9 = df["Clusters"].to_numpy()
y9 = df["Iterations"].to_numpy()
# plotting the line 2 points
plt.plot(x9, y9, label = "random mean Auto")


# df = pd.read_csv("random_mean_auto_m2.csv")
# # line 10 for random mean Auto 2
# x10 = df["Clusters"].to_numpy()
# y10 = df["Iterations"].to_numpy()
# # plotting the line 2 points
# plt.plot(x10, y10, label = "random mean Auto *2")

plt.xlabel('Clusters')
# Set the y axis label of the current axis.
plt.ylabel('Iterations')
# Set a title of the current axes.
plt.title('Comparing Iterations ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()