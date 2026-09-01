def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    if not points:
        return []
        
    num_features = len(points[0])
    centroids = [[0.0] * num_features for _ in range(k)]
    dict_assignments = {i: 0 for i in range(k)}
    for point, assignment in zip(points, assignments):
        dict_assignments[assignment] += 1
        for i in range(num_features):
            centroids[assignment][i] += float(point[i])

    # Compute mean for each centroid
    return [
        [x / dict_assignments[i] for x in centroid] if dict_assignments[i] > 0 else centroid
        for i, centroid in enumerate(centroids)
    ]