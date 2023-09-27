import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv('stars.csv')

wcss = []

for o in range(1,11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state = 42)
  kmeans.fit(list)
  wcss.append(kmeans.inertia_)

kmeans = KMeans(n_clusters=3,init = 'k-means++',random_state=42)
ykmeans = kmeans.fit_predict(list)
plt.figure(figsize = (10,6))
sns.scatterplot(list[y_kmeans == 0,0],list[y_kmeans == 0,1],color = 'red')
sns.scatterplot(list[y_kmeans == 1,0],list[y_kmeans == 1,1])
sns.scatterplot(list[y_kmeans == 2,0],list[y_kmeans == 2,1])
sns.scatterplot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color = 'pink',)
plt.xlabel('Mass')
plt.ylabel('Radius')
plt.show()