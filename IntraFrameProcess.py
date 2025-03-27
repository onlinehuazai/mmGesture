import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

def normalize_coordinates(points):
    scaler = MinMaxScaler()
    return scaler.fit_transform(points)

def statistical_outlier_removal_intra_frame(data, k, std_dev_multiplier):
    if k < 4:
        return data, np.ones(len(data), dtype=bool)
    
    points = np.array(data)
    normalized_points = normalize_coordinates(points)
    
    nbrs = NearestNeighbors(n_neighbors=k, metric='manhattan').fit(normalized_points)
    distances, indices = nbrs.kneighbors(normalized_points)
    distances = distances[:, 1:]
    
    mean_distances = distances.mean(axis=1)
    global_mean = mean_distances.mean()
    global_std_dev = mean_distances.std()
    
    threshold = global_mean + std_dev_multiplier * global_std_dev
    inliers = mean_distances < threshold
    
    return points[inliers].tolist(), inliers