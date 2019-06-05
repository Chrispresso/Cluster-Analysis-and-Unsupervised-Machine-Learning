import numpy as np
import matplotlib.pyplot as plt

def d(u, v):
    diff = u-v
    return diff.dot(diff)

def cost(X, R, M):
    cost = 0
    for k in range(len(M)):
        for n in range(len(X)):
            cost += R[n, k] * d(M[k], X[n])
    return cost

def plot_k_means(X, K, title, max_iter=20, beta=1.0):
    N, D = X.shape
    M = np.zeros((K, D))
    # R = np.zeros((N, K))
    exponents = np.empty((N, K))

    for k in range(K):
        M[k] = X[np.random.choice(N)]

    costs = []

    k = 0
    for i in range(max_iter):
        k += 1
        # step 1: determine assignments / resposibilities
        # is this inefficient?
        for k in range(K):
            for n in range(N):
                exponents[n,k] = np.exp(-beta*d(M[k], X[n]))
        R = exponents / exponents.sum(axis=1, keepdims=True)


        # step 2: recalculate means
        # decent vectorization
        # for k in range(K):
        #     M[k] = R[:,k].dot(X) / R[:,k].sum()
        # oldM = M

        # full vectorization
        M = R.T.dot(X) / R.sum(axis=0, keepdims=True).T
        # print("diff M:", np.abs(M - oldM).sum())

        c = cost(X, R, M)
        costs.append(c)
        if i > 0:
            if np.abs(costs[-1] - costs[-2]) < 1e-5:
                break

        if len(costs) > 1:
            if costs[-1] > costs[-2]:
                pass
                # print("cost increased!")
                # print("M:", M)
                # print("R.min:", R.min(), "R.max:", R.max())

    plt.plot(costs)
    plt.title('Costs ' + title)
    plt.show()

    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.title(title)
    plt.show()

def gaussian_clouds():
    D = 2  # Dimensionality
    s = 4  # Spacing between cloud centers
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900  # total samples - 300/class
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3

    plt.scatter(X[:,0], X[:,1])
    plt.show()

    K = 3
    plot_k_means(X, K, 'K=3')

    K = 5
    plot_k_means(X, K, 'K=5', max_iter=30)

    K = 5
    plot_k_means(X, K, 'K=5, b=.3', max_iter=30, beta=.3)



if __name__ == "__main__":
    gaussian_clouds()