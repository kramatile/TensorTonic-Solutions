def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    def compute_score(item,min_votes):
        return item[1]/(item[1]+min_votes) * item[0]
    scores = []
    for item in items : 
        scores.append(compute_score(item,min_votes) + (min_votes / (min_votes + item[1])*global_mean))
    return scores        