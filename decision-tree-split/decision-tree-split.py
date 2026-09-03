def decision_tree_split(X: list, y: list) -> list:
    """
    Returns the best feature index and threshold.
    """
    n_cols = len(X[0])
    cols_tresholds = {}
    for i in range(n_cols):
        values = sorted([row[i] for row in X])
        tresholds = []
        for l in range(len(values)-1):
            r = l+1
            if values[r] == values[l]:
                continue 
            tresholds.append((values[r]+values[l])/2)
        cols_tresholds[i] = tresholds
        
    import numpy as np
    
    def gini_impurity(y_left: list, y_right: list) -> float:
        """
        Returns the impurity as a float.
        """
        y_left = np.asarray(y_left,dtype=int)
        y_right = np.asarray(y_right,dtype=int)
        
        y_left_cat, y_left_count = np.unique(y_left,return_counts=True)
        y_right_cat, y_right_count = np.unique(y_right,return_counts=True)
    
        left_probas = y_left_count/len(y_left)
        right_probas = y_right_count/len(y_right)
    
        gini_left = 1 - np.sum(left_probas**2)
        gini_right = 1 - np.sum(right_probas**2)
        if len(y_left)+len(y_right) == 0:
            return 0 
        gini =( len(y_left)/(len(y_left)+len(y_right)) *gini_left)
        gini+=( len(y_right)/(len(y_left)+len(y_right)) *gini_right)
        
        return gini   
    min_split_col = None
    min_split_treshold = None
    min_gini = float("inf")
    for col, tresholds in cols_tresholds.items():
        if not tresholds : 
            continue 

        for treshold in tresholds :
            y_left = []
            y_right = []
            for i in range(len(X)): 
                if X[i][col] < treshold : 
                    y_left.append(y[i])
                else : 
                    y_right.append(y[i])
            gini = gini_impurity(y_left,y_right)
            if gini < min_gini:
                min_gini = gini
                min_split_col = col
                min_split_treshold = treshold
            
            
    return [min_split_col, min_split_treshold]