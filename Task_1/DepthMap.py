import numpy as np

# Define SGM parameters
window_size = 5
disparity_range = 64
P1 = 10
P2 = 120

# Step 1: Compute matching cost volume
def compute_cost_volume(left_img, right_img):
    h, w = left_img.shape
    cost_volume = np.zeros((h, w, disparity_range), dtype=np.float32)
    
    for d in range(disparity_range):
        right_shifted = np.roll(right_img, -d, axis=1)
        cost_volume[:, :, d] = np.abs(left_img - right_shifted)
    
    return cost_volume

# Step 2: Aggregate costs using dynamic programming (DP)
def aggregate_costs(cost_volume):
    h, w, d = cost_volume.shape
    aggregated_costs = np.zeros_like(cost_volume)
    aggregated_costs[:, :, 0] = cost_volume[:, :, 0]
    
    for i in range(1, d):
        previous_costs = np.array([aggregated_costs[:, j, i - 1] for j in range(w)])
        min_previous_costs = np.min(previous_costs, axis=0)
        aggregated_costs[:, :, i] = cost_volume[:, :, i] + min_previous_costs
    
    return aggregated_costs

# Step 3: Find disparity map using winner-takes-all (WTA) strategy
def find_disparity_map(aggregated_costs):
    h, w, d = aggregated_costs.shape
    disparity_map = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            disparity_map[i, j] = np.argmin(aggregated_costs[i, j])
    
    return disparity_map

# Step 4: Post-processing (optional)
def post_process(disparity_map):
    return disparity_map

# Read stereo images (assuming grayscale)
left_img = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)

# Step 1: Compute cost volume
cost_volume = compute_cost_volume(left_img, right_img)

# Step 2: Aggregate costs using dynamic programming (DP)
aggregated_costs = aggregate_costs(cost_volume)

# Step 3: Find disparity map using winner-takes-all (WTA) strategy
disparity_map = find_disparity_map(aggregated_costs)

# Step 4: Post-processing (optional)
disparity_map = post_process(disparity_map)

# Save disparity map
cv2.imwrite('disparity_map.png', disparity_map)
