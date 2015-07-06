from kameleon_rks.banana import sample_banana
from kameleon_rks.gaussian import sample_gaussian
from kameleon_rks.gaussian_rks import sample_basis, feature_map,\
    feature_map_single, feature_map_grad_single
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)

# fix basis for now
D = 2
m = 1000
gamma = .3
omega, u = sample_basis(D, m, gamma)

# sample points in input space
N = 2000
Z = sample_banana(N, D)

# fit Gaussian in feature space
Phi = feature_map(Z, omega, u)

# mean and covariance, batch version
mu = np.mean(Phi, 0)
eta = 0.01
C = np.cov(Phi.T) + eta ** 2 * np.eye(m)
L = np.linalg.cholesky(C)

# step size
eta = 50.

plt.plot(Z[:, 0], Z[:, 1], 'bx')

# proposal plotting colors
colors = ['y', 'r', 'g', 'm', 'black']

# proposals centred at those points
Ys = np.array([[-20, 9.7], [-10, 0], [0,-3], [10, 0], [20, 9.7]])

for j in range(len(colors)):

    # pick point at random, embed, gradient
    y = Ys[j]
    phi_y = feature_map_single(y, omega, u)
    grad_phi_y = feature_map_grad_single(y, omega, u)
    
    # draw a number of proposals at the current point
    n_proposals = 100
    X_star = np.zeros((n_proposals, D))
    
    # generate proposal samples in feature space
    for i in range(n_proposals):
        plt.plot(y[0], y[1], '*', markersize=15, color=colors[j])
        
        # construct covariance, adding exploration noise
        R = eta**2 * np.dot(grad_phi_y, np.dot(C, grad_phi_y.T))
        L_R = np.linalg.cholesky(R)
        
        # sample proposal
        x_star = sample_gaussian(N=1, mu=y, Sigma=L_R, is_cholesky=True)
        
        X_star[i] = x_star
    
    plt.plot(X_star[:, 0], X_star[:, 1], 'x', color=colors[j])
plt.show()
