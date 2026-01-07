import numpy as np

def random_simple_graph(n, p):
    A = (np.random.rand(n, n) < p).astype(int)
    A = np.triu(A, 1)
    A = A + A.T
    return A

def graph_invariants(A):
    d = A.sum(axis=1)
    dG = d.sum()
    sum_d2 = np.sum(d**2)
    frob_A2 = np.sum(A**2)   # = dG for simple graphs
    return d, dG, sum_d2, frob_A2

def alpha_ppt_threshold(dG, sum_d2, m, n):
    term = dG/(m*n - 1) - sum_d2/dG
    if term <= 0:
        return 1.0  # always PPT
    return 1 / (1 + np.sqrt(term))

def alpha_positivity(lambda_min_A, delta):
    return lambda_min_A / (lambda_min_A - delta)

def sample_density_matrix(n=9,p=0.3,m=3,n_sub=3):
    A = random_simple_graph(n, p)
    d, dG, sum_d2, frob_A2 = graph_invariants(A)
    if dG == 0:
        return None
    D = np.diag(d)

    delta = d.min()
    lambda_min = np.linalg.eigvalsh(A)[0]
    alpha0 = alpha_positivity(lambda_min, delta)
    if alpha0 >= 1 - 1e-12:
        return None

    alpha_ppt = alpha_ppt_threshold(dG, sum_d2, m, n_sub)

    if np.random.rand() < 0.5 and alpha0 < alpha_ppt:
        alpha = np.random.uniform(alpha0, alpha_ppt)
        label = 1
    else:
        alpha = np.random.uniform(max(alpha_ppt, alpha0), 1)
        label = 0

    M = alpha * D + (1 - alpha)*A
    rho = M / (alpha * dG)

    return rho, label, alpha, alpha_ppt

def gell_mann_matrices():
    l = []

    # λ1
    l.append(np.array([[0,1,0],
                       [1,0,0],
                       [0,0,0]]))

    # λ2
    l.append(np.array([[0,-1j,0],
                       [1j,0,0],
                       [0,0,0]]))

    # λ3
    l.append(np.array([[1,0,0],
                       [0,-1,0],
                       [0,0,0]]))
    # λ4
    l.append(np.array([[0,0,1],
                       [0,0,0],
                       [1,0,0]]))

    # λ5
    l.append(np.array([[0,0,-1j],
                       [0,0,0],
                       [1j,0,0]]))

    # λ6
    l.append(np.array([[0,0,0],
                       [0,0,1],
                       [0,1,0]]))
    # λ7
    l.append(np.array([[0,0,0],
                       [0,0,-1j],
                       [0,1j,0]]))

    # λ8
    l.append((1/np.sqrt(3)) *
             np.array([[1,0,0],
                       [0,1,0],
                       [0,0,-2]]))

    return l
I3 = np.eye(3)
gell_mann = [I3] + gell_mann_matrices()
def gell_mann_features(rho):
    """
    Input:
        rho : 9x9 density matrix (two-qutrit state)

    Output:
        feature vector of length 80
    """
    features = []
    basis = gell_mann

    for i in range(9):
        for j in range(9):
            if i == 0 and j == 0:
                continue  # skip identity ⊗ identity
            op = np.kron(basis[i], basis[j])
            val = np.trace(rho @ op).real
            features.append(val)

    return np.array(features)







    
    
