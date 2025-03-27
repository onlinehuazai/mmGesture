from scipy.spatial import distance
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_coordinates(points):
    scaler = MinMaxScaler()
    return scaler.fit_transform(points)

def calculate_distance_sum(points):
    distances = []
    n = len(points)
    normalized_points = normalize_coordinates(points)

    for i in range(n):
        distance_sum = 0

        if i == 0:
            distance_sum += distance.cityblock(normalized_points[0], normalized_points[1])
        if i > 0:
            distance_sum += distance.cityblock(normalized_points[i-1], normalized_points[i])
        if i < n - 1:
            distance_sum += distance.cityblock(normalized_points[i], normalized_points[i + 1])
        if i == n-1:
            distance_sum += distance.cityblock(normalized_points[n-2], normalized_points[n-1])
        
        distances.append(distance_sum)
    return np.array(distances)

def statistical_outlier_removal_inter_frame(data, std_dev_multiplier):
    points = np.array(data)
    distance_sums = calculate_distance_sum(points)
    mean_distance = distance_sums.mean()
    std_dev_distance = distance_sums.std()
    threshold_up = mean_distance + std_dev_multiplier * std_dev_distance
    threshold_low = mean_distance - std_dev_multiplier * std_dev_distance
    inliers = (distance_sums < threshold_up) & (distance_sums > threshold_low)
    return points[inliers].tolist()