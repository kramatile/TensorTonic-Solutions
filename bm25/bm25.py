import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    if not docs :
        return np.array([])
    def df(term: str, docs: list[list[str]]) -> float:
        per_doc_count = [1 if term in l else 0 for l in docs]
        return sum(per_doc_count)
        
    def idf(terms: list[str],docs: list[list[str]]) -> np.ndarray:
        N = len(docs)
        dfs = np.array([df(term,docs) for term in terms],dtype=np.float64)
        
        return np.log(((N -dfs+0.5)/(dfs+0.5))+1)
        
    def tf(term: str, doc: list[str])-> float:
        return doc.count(term)

    # build tf matrix of shape (number of)
    tfs : np.ndarray = np.zeros((len(query_tokens),len(docs)))
    for i,term in enumerate(query_tokens):
        for j,doc in enumerate(docs):
            tfs[i][j] = tf(term,doc)
    idfs : np.ndarray = idf(query_tokens,docs)
    avgdl : float = sum([len(doc) for doc in docs])/len(docs)
    denominateur : np.ndarray = np.array([k1*(1-b+b*(len(doc)/avgdl)) for doc in docs])
    return (((tfs * (k1+1))/(tfs+denominateur))*idfs.reshape(len(idfs),1)).sum(axis=0)