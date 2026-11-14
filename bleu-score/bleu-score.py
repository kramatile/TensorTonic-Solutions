import math
from collections import Counter

def bleu_score(candidate: list, reference: list, max_n: int) -> float:
    """
    Returns the unsmoothed BLEU score.
    """
    if not candidate : 
        return 0.0
    def generate_n_grams(sentence: list[str], n: int) -> list[tuple[str]] :
        n_grams = []
        for i in range(0,len(sentence)-n+1):
            n_grams.append(tuple(sentence[i:i+n]))
        return n_grams
            
    def prenalties_n_grams(candidate: list, reference : list, ns : int) -> list[float]:
        penalties = []
        for n in range(1,ns+1):
            reference_n_grams = Counter(generate_n_grams(reference,n))
            candidate_n_grams = Counter(generate_n_grams(candidate,n))
            penalty = 0.0
            den = 0.0
            print(generate_n_grams(candidate,n))
            for k, v in candidate_n_grams.items():
                penalty += min(v, reference_n_grams.get(k,0))
                den += v
            penalties.append(penalty/den)
        return penalties

    def brevity_penalty(r_length: int, c_length:int) -> float:
        return 1 if c_length >= r_length else math.exp(1-(r_length/c_length))

    penalties = prenalties_n_grams(candidate,reference,max_n)
    geometric_mean = 0.0
    for penalty in penalties:
        if penalty == 0.0 : 
            return 0.0
        geometric_mean += math.log(penalty)
    geometric_mean = math.exp(geometric_mean/max_n)
    return brevity_penalty(len(reference),len(candidate)) * geometric_mean