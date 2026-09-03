import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """  
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if X_test.size == 0:
        return np.empty((0, k), dtype=int)
    if X_train.size == 0:
        return np.empty((0, k), dtype=int)

    if X_train.ndim == 1:
        X_train = X_train[:, np.newaxis]
    if X_test.ndim == 1:
        X_test = X_test[:, np.newaxis]

    arg_distances = []
    for x_test in X_test: 
        distances = (X_train - x_test)**2
        if X_train.ndim == 2: 
            distances = np.sum(distances, axis=1)
        
        arg_sort = np.argsort(distances)[:k].tolist()
        
        while len(arg_sort) < k:
            arg_sort.append(-1)
            
        arg_distances.append(arg_sort)
        
    return np.array(arg_distances, dtype=int)