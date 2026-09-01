def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    assignments = []
    for point in points : 
        min_distance = float("inf")
        index_min_distance = None 
        i = 0
        for centroid in centroids : 
            distance = 0
            for x,y in zip(centroid,point):
                distance += (x-y)**2 
            if distance < min_distance : 
                min_distance = distance
                index_min_distance = i
            i += 1
        assignments.append(index_min_distance)       
    return assignments