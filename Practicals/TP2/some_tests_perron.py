import numpy as np
import matplotlib.pyplot as plt

def construct_leslie(N, s_list, f_list):
    if (len(f_list) != N) or (len(s_list) != N-1):
        raise("ValueError")
    L = np.zeros((N, N))
    for j in range(N):
        L[0, j] = f_list[j]
    for i in range(N-1):
        L[i+1, i] = s_list[i]
    return L


N_a1 = 3
s_list_a1 = [0.25, 0.5]
f_list_a1 = [0, 0, 8]
les_a1 = construct_leslie(N_a1, s_list_a1, f_list_a1)
print(les_a1)

X = [1, 1, 1]
X_vals = []

iter = 0
max_iter = 4

iters = [_ for _ in range(max_iter)]

while iter < max_iter:
    X = les_a1 @ X
    X_vals.append(X)
    iter += 1

X_vals = np.array(X_vals)

xs = X_vals[:, 0]
ys = X_vals[:, 1]
zs = X_vals[:, 2]

fig, axs = plt.subplots(1, 3)
ax1, ax2, ax3 = axs
plt.style.use('ggplot')
ax1.plot(iters, xs, label='class1')
ax2.plot(iters, ys, label='class2')
ax3.plot(iters, zs, label='class3')
plt.show()
