from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
from sklearn.cluster import DBSCAN

X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
df=pd.DataFrame(X,columns=["X","Y"])

plt.scatter(df["X"],df["Y"])
plt.title("Moon Dataset")
plt.show()

kmeans=KMeans(n_clusters=2,random_state=42)
cluster=kmeans.fit_predict(df)
df["cluster"]=cluster
sns.scatterplot(data=df,x=df["X"],y=df["Y"],hue=cluster)
plt.title("Using Kmeans Clustering")
plt.show()


db=DBSCAN(eps=0.2,min_samples=4)
db_cluster=db.fit_predict(df[["X","Y"]])
df["db_cluster"]=db_cluster
sns.scatterplot(data=df,x=df["X"],y=df["Y"],hue="db_cluster")
plt.title("Using DBSCAN")
plt.show()

