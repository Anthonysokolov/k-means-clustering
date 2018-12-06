"""
Basic K Means Clustering Algorithm

Algorithm:
    1. Choose k random datapoints to act as initial centroids
    2. Assign each datapoint to the cluster of the nearest centroid
    3. Recalculate the centroids by finding the average value of every cluster
    4. Repeat until convergence
    
Data:
    2013 NBA Point Guard Stats (via DataQuest)
"""
import random
import matplotlib.pyplot as plt
import pandas as pd

def kmeans_cluster(k,data):
    '''
    Divides a set of data into k clusters based on proximity
    '''
    # Choose random datapoints as initial clusters
    centroid_ids = generate_ids(k,data)
    # Store the centroid values in a dict
    centroid_dict = generate_centroid_dict(centroid_ids,data)
        
    old_dict = {}    
    cols = data.columns

    while centroid_dict != old_dict:
        clusters = []
        # Assign each datapoint to the cluster with the nearest centroid
        for i in range(len(data)):
            clusters.append(assign_cluster(centroid_dict,
                                           data.loc[data.index[i],cols].tolist()))
            
        # Recalculate centroids and repeat until the optimal centroids are found
        data['cluster'] = clusters
        old_dict = centroid_dict
        centroid_dict = update_centroids(centroid_dict,data)
        
    return data


def euclidean_distance(a,b):
    '''
    Calculates the euclidean distance between two arrays
    '''
    total = 0
    for x1,x2 in zip(a,b):
        total += (x1-x2)**2
        
    return total**.5


def generate_ids(k,df):
    '''
    Chooses k random points as the initial centroids
    '''
    centroid_ids = []
    for i in range(k):
        c = random.choice(range(len(df)))
        centroid_ids.append(c)
    if centroid_ids == list(set(centroid_ids)):
        return centroid_ids
    else:
        return generate_ids(k,df)
        
    

def generate_centroid_dict(ids, df):
    '''
    Creates a dict containing centroid labels and their values
    given their indices
    '''
    centroid_dict = {}
    x = 0
    for i in ids:
        centroid_dict[x] = df.iloc[i,:].tolist()
        x += 1
        
    return centroid_dict
    


def update_centroids(old_centroids,df):
    '''
    Updates the value of each centroid to the average of all values in 
    its respective cluster
    '''
    new_centroids = {}
    
    for k in old_centroids:
        cluster = df[df['cluster']==k]
        cluster = cluster.drop(columns=['cluster'])
        new_centroids[k] = cluster.mean().tolist()
        
        
    
    return new_centroids


def assign_cluster(centroids, row):
    '''
    Assigns a datapoint to the cluster of the nearest centroid
    '''
    keys = list(centroids.keys())
    lowd = euclidean_distance(centroids[keys[0]],row)
    cluster = keys[0]
    
    for c in keys[1:]:
        d = euclidean_distance(centroids[c],row)
        if d < lowd:
            lowd = d
            cluster = c

    return cluster
    

# Read in csv file
df = pd.read_csv('nba_2013.csv')
# Filter out any point guards who played less than 5 games
df = df[df['gs'] > 5]
# Create a column measuring points per game
df['ppg'] = df['pts']/df['gs']
# Create a column measuring assist to turnover ratios
df['atr'] = df['ast']/df['tov']

# Normalize the data
df['ppg'] = df['ppg']/df['ppg'].mean()
df['atr'] = df['atr']/df['atr'].mean()

# Assign clusters and plot the data
train = df.loc[:,['ppg','atr']]
df = kmeans_cluster(5,train)

for i,color in enumerate(['red','blue','yellow','black','green']):
    cluster = train[train['cluster'] == i]
    plt.scatter(cluster['atr'],cluster['ppg'],c=color)


































