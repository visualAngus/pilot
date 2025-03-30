def enhance_contrast(depth_map, lower_percentile=2, upper_percentile=98):
    """Enhance the contrast of the depth map using percentile scaling."""
    lower_bound = np.percentile(depth_map, lower_percentile)
    upper_bound = np.percentile(depth_map, upper_percentile)
    
    # Clip the depth map to the specified bounds
    depth_map_clipped = np.clip(depth_map, lower_bound, upper_bound)
    
    # Normalize the depth map to the range [0, 255]
    depth_map_normalized = (depth_map_clipped - depth_map_clipped.min()) / (depth_map_clipped.max() - depth_map_clipped.min()) * 255
    return depth_map_normalized.astype(np.uint8)

def select_distance_range(depth_map, min_distance, max_distance):
    """Select a specific distance range from the depth map."""
    # Create a mask for the specified distance range
    mask = (depth_map >= min_distance) & (depth_map <= max_distance)
    selected_depth_map = np.zeros_like(depth_map)
    selected_depth_map[mask] = depth_map[mask]
    return selected_depth_map