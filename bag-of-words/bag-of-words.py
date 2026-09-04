import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    bow = [0 for _ in range(len(vocab))]
    vocab_dict = {word: i for i, word in enumerate(vocab)}
    for token in tokens :
        if token in vocab_dict:
            bow[vocab_dict[token]] += 1
    return np.array(bow)
            
