#!/usr/bin/env python3

"""
Made With <3 By Théo
"""

#imports
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


#functions
def get_svd_reduction(M, k=50):
    U, S, V = la.svd(M)
    S = np.diag(S)
    A_k = U[:, :k] @ S[0:k, :k] @ V[:k, :]
    return A_k


# global parameters
k = 50 # reasonable resolution for both images
fig, ax = plt.subplots(nrows=2, ncols=2) # defines the four subplots


if __name__ == "__main__":
    # image Lena
    img1 = np.asarray(plt.imread("lena512.png"))
    A1_k = get_svd_reduction(img1, k)
    ax[0, 0].imshow(img1, cmap=plt.cm.gray)
    ax[0, 0].set_title("Before Reduction")
    ax[0, 1].imshow(A1_k, cmap=plt.cm.gray)
    ax[0, 1].set_title("After Reduction")

    # image baboon
    img2 = plt.imread("baboon-grayscale.jpg")
    img2 = np.mean(img2, -1)
    print(img2.shape)
    A2_k = get_svd_reduction(img2, k)
    ax[1, 0].imshow(img2, cmap=plt.cm.gray)
    ax[1, 0].set_title("Before Reduction")
    ax[1, 1].imshow(A2_k, cmap=plt.cm.gray)
    ax[1, 1].set_title("After Reduction")

    # plotting everything :D
    plt.show()
