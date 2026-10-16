import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
def relu(x):
    return np.maximum(0,x)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    mu = np.mean(x,axis=-1,keepdims = True)
    var = np.var(x,axis=-1,keepdims = True)
    x_scaled = (x - mu)/np.sqrt(var+eps)
    print(x_scaled)
    return gamma*x_scaled + beta
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    return np.dot(relu(np.dot(x,W1)+b1),W2) + b2

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model//num_heads

    Q = np.dot(Q, W_q)
    K = np.dot(K, W_k)
    V = np.dot(V, W_v)
    
    Q = Q.reshape(batch_size,seq_len,num_heads,d_k).transpose(0,2,1,3)
    K = K.reshape(batch_size,seq_len,num_heads,d_k).transpose(0,2,1,3)
    V = V.reshape(batch_size,seq_len,num_heads,d_k).transpose(0,2,1,3)

    scores = np.matmul(Q,K.transpose(0,1,3,2))/np.sqrt(d_k)
    head_weights = softmax(scores)
    attention_output = np.matmul(head_weights,V)

    attention_output = attention_output.transpose(0,2,1,3).reshape(batch_size,seq_len,d_model)
    return np.matmul(attention_output,W_o)
def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    mha = x+multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    z = layer_norm(mha,gamma1,beta1)
    return layer_norm(z+feed_forward(z,W1,b1,W2,b2),gamma2,beta2)
    