def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    # Write code here
    union_recs = set()
    for rec in recommendations : 
        union_recs = union_recs | set(rec)
    return len(union_recs)/n_items