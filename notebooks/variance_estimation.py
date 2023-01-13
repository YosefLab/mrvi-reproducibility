# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
A = np.random.randn(10, 10) * 10
std = np.random.randn(1, 10, 1)

# %%
# Monte carlo simulation of l2 distances between gaussian samples
u = np.random.randn(20000, 10, 1) * std
u2 = np.random.randn(20000, 10, 1) * std
z, z2 = u + np.einsum('ij, cjk->cik', A, u), u2 + np.einsum('ij,cjk->cik', A, u2)
mc_dists = np.linalg.norm(z - z2, axis=1)

# %%
# Partial analytic computation with monte carlo
sigma = np.diag(std.flatten()**2)
eigval, _ = np.linalg.eig(2 * (np.eye(A.shape[0]) + A) @ sigma @ (np.eye(A.shape[0]) + A).T)
mc_samples = np.random.randn(20000, 10)
mc_squared_l2 = np.sum((mc_samples ** 2) * eigval.reshape(1, -1), axis=1)
mc_dists_partial = mc_squared_l2 ** 0.5

# %%
plt.hist(mc_dists, bins=100, label="Full Monte Carlo", alpha=0.5)
plt.hist(mc_dists_partial, bins=100, label="Partial Monte Carlo", alpha=0.5)
plt.legend()

# %%
print(f"F Mean: {np.mean(mc_dists)}")
print(f"F Std: {np.std(mc_dists)}")
print(f"P Mean: {np.mean(mc_dists_partial)}")
print(f"P Std: {np.std(mc_dists_partial)}")

# %%
