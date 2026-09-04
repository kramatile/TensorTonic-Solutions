def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    step = chunk_size - overlap
    text_chunks = []
    for i in range(0,len(tokens),step):
        text_chunks.append(tokens[i:i+chunk_size])
        if i + chunk_size >= len(tokens):
            break
    return text_chunks
        