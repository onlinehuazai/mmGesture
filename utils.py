import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import cv2

def clutterRemoval(input_val, axis=1):
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)
    
    mean = input_val.mean(0)
    output_val = input_val - np.expand_dims(mean, axis=0)
    return output_val.transpose(reordering)

def MediaFilter(data, k=2, threshold=0.02, mindistance=0.05):
    filtered_data = []
    
    for i in range(len(data)):
        start = max(0, i - k)
        end = min(len(data), i + k + 1)
        window_points = data[start:end]
        
        median_x = np.median([point[0] for point in window_points])
        median_y = np.median([point[1] for point in window_points])
        current_point = data[i]
        distance = np.sqrt((current_point[0] - median_x)**2 + (current_point[1] - median_y)**2)

        if i < k or len(data) - i - 1 < k:
            filtered_data.append(current_point)
        elif distance <= threshold:
            filtered_data.append(current_point)

    x_list = [point[0] for point in filtered_data]
    y_list = [point[1] for point in filtered_data]

    i = 0
    while i < len(x_list) - 1:
        x1, y1 = x_list[i], y_list[i]
        x2, y2 = x_list[i+1], y_list[i+1]
        distance1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if distance1 > mindistance:
            mid_x = (x1 + x2)/2
            mid_y = (y1 + y2)/2
            x_list.insert(i+1, mid_x)
            y_list.insert(i+1, mid_y)
            i -= 1
        i += 1

    tmpx, tmpy = 0.5, 0.5
    for i in range(1, len(x_list)):
        x_list[i] = tmpx*x_list[i-1] + (1-tmpx)*x_list[i]
        y_list[i] = tmpy*y_list[i-1] + (1-tmpy)*y_list[i]

    return x_list, y_list

def adjust_x_values(x_list, y_list, factor):
    threshold = sum(y_list)/len(y_list)
    max_value = max(y_list)
    min_value = min(y_list)
    adjusted_x_list = []
    
    for x, y in zip(x_list, y_list):
        if y > threshold:
            cur_factor = (y-threshold)/(max_value-threshold)*factor
            adjusted_x = x * (cur_factor+1)
        else:
            cur_factor = (threshold-y)/(threshold-min_value)*factor
            adjusted_x = x * (1-cur_factor)
            
        adjusted_x_list.append(adjusted_x)
    
    return adjusted_x_list, y_list

def kalman_filter_on_lists(x_list, y_list, delta_t=1, Q=0.05, R=0.01):
    n = len(x_list)
    filtered_x_list = []
    filtered_y_list = []

    A = np.array([
        [1, 0, delta_t, 0],
        [0, 1, 0, delta_t],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    X = np.zeros((4, 1))
    P = np.eye(4)
    Qk = Q * np.eye(4)
    Rk = R * np.eye(2)
    
    for i in range(n):
        Z = np.array([x_list[i], y_list[i]]).reshape(-1, 1)
        X = A @ X
        P = A @ P @ A.T + Qk
        
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rk)
        X = X + K @ (Z - H @ X)
        P = (np.eye(4) - K @ H) @ P
        
        filtered_x_list.append(X[0, 0])
        filtered_y_list.append(X[1, 0])

    return filtered_x_list, filtered_y_list

def particle_filter(x_list, y_list, num_particles, num_steps, process_noise_cov, measurement_noise_cov):
    particles = np.zeros((num_particles, 2))
    weights = np.ones(num_particles) / num_particles
    indices = np.random.choice(len(x_list), num_particles, replace=True)
    particles[:, 0] = [x_list[i] for i in indices]
    particles[:, 1] = [y_list[i] for i in indices]
    
    estimated_trajectory = []

    for t in range(num_steps):
        process_noise = np.random.multivariate_normal([0, 0], process_noise_cov, num_particles)
        particles += process_noise

        if t < len(x_list):
            measurement = np.array([x_list[t], y_list[t]])
            measurement_noise = np.random.multivariate_normal([0, 0], measurement_noise_cov, num_particles)
            predicted_measurements = particles + measurement_noise
            distances = np.linalg.norm(predicted_measurements - measurement, axis=1)
            weights = np.exp(-0.5 * (distances**2 / np.linalg.norm(measurement_noise_cov)))
            weights /= np.sum(weights)

        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.
        random_values = np.random.rand(num_particles)
        indices = np.searchsorted(cumulative_sum, random_values)
        particles = particles[indices]
        weights.fill(1.0/num_particles)

        estimated_state = np.mean(particles, axis=0)
        estimated_trajectory.append(estimated_state)

    return estimated_trajectory

def smooth_savgol_filter(x_coords, y_coords, window_length, polyorder):
    x_smooth = savgol_filter(x_coords, window_length, polyorder, mode="mirror")
    y_smooth = savgol_filter(y_coords, window_length, polyorder, mode="mirror")
    return x_smooth, y_smooth

def fill_and_save_image(input_image_path, output_image_path, long_edge_length, resized_size):
    img = cv2.imread(input_image_path)
    if img is None:
        print("Error: Unable to load image.")
        return

    height, width = img.shape[:2]
    scale = long_edge_length / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img_resized = cv2.resize(img, (new_width, new_height))

    background = np.ones((resized_size, resized_size, 3), dtype=np.uint8) * 0
    y_offset = (resized_size - new_height) // 2
    x_offset = (resized_size - new_width) // 2
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_resized

    cv2.imwrite(output_image_path, background, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def plot_trajectory(x_coords, y_coords, image_path, padded_image_path, long_edge=24):
    x = np.array(x_coords)
    y = np.array(y_coords)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    width = x_max - x_min
    height = y_max - y_min

    x_min -= 0.1*width
    x_max += 0.1*width
    y_max += 0.1*height
    y_min -= 0.1*height

    if width > height:
        short_edge = (height/width)*long_edge
        scale_y = short_edge
        scale_x = long_edge
    else:
        short_edge = (width/height)*long_edge
        scale_y = long_edge
        scale_x = short_edge

    scaled_x = (x-x_min)/(x_max-x_min)*scale_x
    scaled_y = (y-y_min)/(y_max-y_min)*scale_y

    if width > height:
        fig, ax = plt.subplots(figsize=(long_edge/256, short_edge/256), dpi=256)
        ax.set_xlim(0, long_edge)
        ax.set_ylim(0, short_edge)
    else:
        fig, ax = plt.subplots(figsize=(short_edge/256, long_edge/256), dpi=256)
        ax.set_xlim(0, short_edge)
        ax.set_ylim(0, long_edge)

    ax.plot(scaled_x, scaled_y, color='white', linewidth=3)
    ax.axis('off')
    
    # Save original trajectory image
    plt.savefig(image_path+'trajectory.png', bbox_inches='tight', pad_inches=0.01, dpi=100, facecolor='black')
    plt.close(fig)
    print(f"Trajectory image successfully saved to: {image_path}trajectory.png")
    # Save resized image
    fill_and_save_image(image_path+'trajectory.png', padded_image_path+'trajectory_resized.png', 
                       long_edge_length=196, resized_size=224)
    print(f"Resized image successfully saved to: {padded_image_path}trajectory_resized.png")