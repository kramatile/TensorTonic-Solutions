import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=int)
    if not max_len:
        max_len = len(max(seqs, key=lambda seq : len(seq)))
    print(max_len)
    for i,seq in enumerate(seqs) : 
        if len(seq) < max_len : 
            while len(seq) < max_len: 
                seq.append(pad_value)
        elif len(seq) > max_len : 
            seqs[i] = seq[:max_len]

    print(seqs)
    return np.array(seqs)