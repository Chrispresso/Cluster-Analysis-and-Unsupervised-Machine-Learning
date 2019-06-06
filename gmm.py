import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def gmm(X, K, max_iter=20, smoothing=1e-2):
    N, D = X.shape
    M = np.zeros((K, D))  # Means
    R = np.zeros((N, K))  # Responsibilities
    C = np.zeros((K, D, D))  # Covarience
    pi = np.ones(K) / K  # Prob

    for k in range(K):
        M[k] = X[np.random.choice(N)]
        C[k] = np.eye(D)

    lls = []
    weighted_pdfs = np.zeros((N, K))

    for i in range(max_iter):
        # Step 1. Determine assignments / responsibilities
        for k in range(K):
            weighted_pdfs[:,k] = pi[k]*multivariate_normal.pdf(X, M[k], C[k])
        R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

        # Step 2. Recalculate params
        for k in range(K):
            Nk = R[:,k].sum()
            pi[k] = Nk / N
            M[k] = R[:,k].dot(X) / Nk

            delta = X - M[k]
            Rdelta = np.expand_dims(R[:,k], -1) * delta 
            C[k] = Rdelta.T.dot(delta) / Nk + np.eye(D)*smoothing 


        ll = np.log(weighted_pdfs.sum(axis=1)).sum()
        lls.append(ll)
        if i > 0:
            if np.abs(lls[i] - lls[i-1]) < 0.1:
                break

    plt.plot(lls)
    plt.title("Log-Likelihood")
    plt.show()

    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.show()

    print("pi:", pi)
    print("means:", M)
    print("covariances:", C)
    return R

def main():
    # assume 3 means
    D = 2  # Dimensions
    s = 4  # Seperation
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 2000
    X = np.zeros((N, D))
    X[:1200, :] = np.random.randn(1200, D)*2 + mu1
    X[1200:1800, :] = np.random.randn(600, D) + mu2
    X[1800:, :] = np.random.randn(200, D)*.5 + mu3

    plt.scatter(X[:,0], X[:,1])
    plt.show()

    K = 3
    gmm(X, K)


if __name__ == "__main__":
    main()