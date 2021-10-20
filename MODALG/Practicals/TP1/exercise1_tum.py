#!/usr/bin/env python3

"""
Made With <3 By Théo
"""

# imports
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats as ss


# functions
def gradient_method(A, ys, t):
    # init "constants"
    d = 2 # A.T @ T 's shape
    A_ = np.dot(A.T, A)
    b_ = np.dot(A.T, ys)
    C_t = np.identity(d) - 2*t*A_
    b_t = 2*t*b_
    residues = []

    # init the loop
    x_old = (0, 0)
    i = 0
    max_iter = 100000
    res = 10

    # looping
    while (i < max_iter and res >= 1e-2):
        x_new = np.dot(C_t, x_old) + b_t
        res = la.norm(np.dot(A_,x_new) - b_)
        residues.append(res)
        x_old = x_new
        i += 1

    # prints the output
    if (i != max_iter):
        print(f"converged in {i} steps")
        return x_new, residues
    else:
        print("haven't converged")
        return residues

# creates the A matrix in J(X) = ||AX-b||^2
# you can increase the degree of interpolation d if needed
def create_A_matrix(t_sample, d=2):
    # d = degree of polynomial of interpolation + 1, here 2 = line
    n = len(t_sample)
    A = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            A[i][j] = t_sample[i]**j
    return A


# the core of the program
def main():
    # input given
    ts = np.array([6, 9, 13, 16, 20, 23, 27, 30, 34, 37])
    ys = np.array([18.76, 19.84, 21.44, 22.19, 22.78, 22.92, 23.43, 23.85, 24.04, 24.38])

    # level of interpolation
    d = 2

    K = 0.5
    y0 = ys[0]
    a = 0.2


"""
    # display data
    #plt.scatter(ts, ys, c='red', marker="o")
    #plt.show()

    # creation of A matrix
    A = create_A_matrix(ts, d)

    # setup for normal equations
    A_ = np.dot(A.T, A)
    b_ = np.dot(A.T, ys)

    # solve normal equations
    X = la.solve(A_, b_)
    # display found solution
    #plt.scatter(ts, ys, marker="o", c="red")
    #lin_data = [X[0] + X[1]*t for t in ts]
    #data = np.exp(lin_data)
    #plt.plot(ts, data, c="blue")
    #plt.show()

    # linear regression via scipy
    (a, b, rho, p, stderr) = ss.linregress(ts, ys)
    #plt.plot(ts, np.exp(np.array([b + a*t for t in ts])), c="blue")
    #plt.scatter(ts, ys, marker="o", c="red")
    #plt.show()

    # least square method
    (x, res, r, s) = la.lstsq(A, ys, rcond=None)
    plt.scatter(ts, ys, marker="o", c="red")
    plt.plot(ts, [x[0] + x[1]*t for t in ts], c="blue")
    plt.show()

    # gradient method
    sol, res1 = gradient_method(A, ys, t=1e-4) # 6072 iterations
    # plot evolution of residues
    #plt.plot([t for t in range(len(res1))], res1)
    #plt.show()

    # optimizing the gradient method using optimal t value
    u, s, v = la.svd(A)
    best_t = 1 / (s[0]**2 + s[1]**2)
    sol2, res2 = gradient_method(A, ys, t=best_t) # converges in 1011 steps

    # plot evolution of best t residues
    #plt.plot([t for t in range(len(res2))], res2)
    #plt.show()
"""

# main program
if __name__ == "__main__":
    main()
