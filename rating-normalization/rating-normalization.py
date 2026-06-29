import numpy as np
def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    normalized_matrix = []
    for user in matrix :
        filtered_user = [rating for rating in user if rating != 0]
        if not filtered_user : 
            normalized_matrix.append(user)
            continue 
        average = sum(filtered_user)/len(filtered_user)
        mean_centered_user = [rating - average if rating != 0 else 0 for rating in user]
        normalized_matrix.append(mean_centered_user)    
    return normalized_matrix