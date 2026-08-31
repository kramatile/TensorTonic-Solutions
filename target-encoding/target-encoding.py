def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    cat_dict = {}
    for cat,tar in zip(categories,targets):
        if cat not in cat_dict :
            cat_dict[cat] = {}
            cat_dict[cat]['count'] = 1.0
            cat_dict[cat]["running_sum"] = tar
        else : 
            cat_dict[cat]['count'] +=1.0
            cat_dict[cat]["running_sum"] += tar

    t_encoding = [cat_dict[cat]["running_sum"]/cat_dict[cat]['count'] for cat in categories]
    return t_encoding