import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    documents = np.asarray(documents)
    N = len(documents)
    vocabulary = []
    vocab_counts = {}
    
    for i in range(len(documents)):
        vocabulary.extend(documents[i].split())
        seen = set()
        for token_ in documents[i].split():
            if token_ in seen:
                continue 
            if token_ not in vocab_counts:
                vocab_counts[token_] = 1
            else : 
                vocab_counts[token_] += 1
            seen.add(token_)
   
             
    vocabulary = np.unique(vocabulary)
    
    tfidf = np.empty((len(documents),len(vocabulary)),float)
    for i in range(len(documents)):
        d_vocab,d_counts = np.unique(documents[i].split(),return_counts = True)
        d_vocab_counts = {vocab: count for vocab, count in zip(d_vocab,d_counts)}
        print(d_vocab_counts)
        tf = np.zeros((len(vocabulary)),float)
        idf = np.zeros((len(vocabulary)),float)
        for j in range(len(vocabulary)): 
            idf[j] = np.log(N/vocab_counts[vocabulary[j]])
            if vocabulary[j] in d_vocab_counts : 
                tf[j] = d_vocab_counts[vocabulary[j]]/len(documents[i].split())
            else :
                tf[j] = 0.0
        print(tf.dtype, idf.dtype)
        tfidf[i] = tf*idf

        
    print(tfidf) 
        
    return {"vocabulary":vocabulary.tolist(),"tfidf_matrix":tfidf}