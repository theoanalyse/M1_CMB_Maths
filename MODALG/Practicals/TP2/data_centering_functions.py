import numpy as np

def get_mean(M):
    return np.sum(M / len(M))

def get_std(M):
    mean = get_mean(M)
    num = np.sum([(x-mean)**2 for x in M])
    std = num / (len(M))
    return np.sqrt(std)

def normalize_data(M):
    rows, cols = M.shape
    new_mat = np.zeros((rows, cols))
    tmp = np.zeros(rows)
    for col in range(cols):
        mean = get_mean(M[:, col])
        sigma = get_std(M[:, col])
        tmp = np.array([])
        for e in M[:, col]:
            tmp = np.append(tmp, (e - mean) / sigma)
        new_mat[:, col] = tmp
    return new_mat
