#!/usr/bin/env python

# IMPORTS
import matplotlib.pyplot as plt
import numpy as np


# FUNCTIONS
def get_spectral_ray_and_its_eigenvector(M):
    sp, eiv = np.linalg.eig(M)
    rho_index = np.argmin(-np.abs(sp))
    rho = sp[rho_index]
    v = eiv[rho_index]
    return rho, v


def construct_leslie(N, s_list, f_list):
    if (len(f_list) != N) or (len(s_list) != N-1):
        raise("ValueError")
    L = np.zeros((N, N))
    for j in range(N):
        L[0, j] = f_list[j]
    for i in range(N-1):
        L[i+1, i] = s_list[i]
    return L


def is_positive(M):
    for i in range(len(M)):
        for j in range(len(M[i])):
            if M[i, j] <= 0:
                return False
    return True


def is_primitive(A):
    """Assuming A is a square matrix, we use the following criterion :

    if A \in M_n(R) is positive. Then A is irreductible iff
    the matrix (I_n + A)^{n-1} is positive

    """
    n = len(A)
    I = np.eye(n)
    mat = np.linalg.matrix_power(I+A, n-1)
    if is_positive(mat):
        return "This matrix is primitive"
    return "This matrix is not primitive"


def process_iterations(L):
    """
    input : L, a matrix such that X^n+1 = LX
    output: X^n, Y^n, (LY, Y)
    """
    n = len(L)
    err = 1
    iter = 0
    max_iter = 1e5
    X_start = [1 for _ in range(n)]
    while (iter < max_iter and err > 1e-12 and err < 1e12):
        X_n = L @ X_start
        err = np.linalg.norm(X_n - X_start)
        X_start = X_n
        iter += 1

    L_n = np.linalg.matrix_power(L, iter)
    Y_n = ( L_n @ X_start) / np.linalg.norm(L_n @ X_start)
    LY_dot_Y = (L @ Y_n).T @ Y_n
    print(f"converged in {iter} iterations")
    return X_n, Y_n, LY_dot_Y

# MAIN PROGRAM
# Academic 1
N_a1 = 3
s_list_a1 = [0.5, 0.5]
f_list_a1 = [0, 1/2, 1]
les_a1 = construct_leslie(N_a1, s_list_a1, f_list_a1)
rho, v = get_spectral_ray_and_its_eigenvector(les_a1)
b, y, r = process_iterations(les_a1)


# Academic 2
N_a2 = 4
s_list_a2 = [0.5, 0.5, 0.5]
f_list_a2 = [0, 0, 0, 1]
les_a2 = construct_leslie(N_a2, s_list_a2, f_list_a2)


# My attempt
N_t = 10
s_list_t = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
f_list_t = [0.1 for _ in range(10)]
les_t = construct_leslie(N_t, s_list_t, f_list_t)


# Black Footed Ferrets
N_f = 5
s_list_f = [0.39, 0.67, 0.67, 0.67]
f_list_f = [0.73, 1.25, 1.25, 1.25, 0]
les_f = construct_leslie(N_f, s_list_f, f_list_f)


# Tibetan Monkey
N_m = 7
s_list_m = [0.803, 0.802, 0.868, 0.868, 0.868, 0.868]
f_list_m = [0, 0, 3.124, 3.124, 3.124, 3.124, 3.124]
les_m = construct_leslie(N_m, s_list_m, f_list_m)


# Question 1 : Check Primitivity of all matrices
mats = [les_a1, les_a2, les_t, les_f, les_m]
for mat in mats:
    print(is_primitive(mat)) # Expected : True True (True/False) False True


# Question 2 : Illustrate Perron-Frobenius Theorem

# Question 3 interprete results
