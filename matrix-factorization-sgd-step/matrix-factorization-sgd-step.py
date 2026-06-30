def matrix_factorization_sgd_step(U, V, r, lr, reg):
    """
    Perform one SGD step for matrix factorization.
    """
    # Write code here
    dot_product = 0
    for x,y in zip(U,V):
        dot_product +=(x*y)
    e = r - dot_product
    U_new = U.copy()
    V_new = V.copy()
    for i in range(len(U)): 
        U_new[i] = U[i] + lr*(e*V[i] - reg*U[i])
        V_new[i] = V[i] + lr*(e*U[i] - reg*V[i])
    return (U_new,V_new)