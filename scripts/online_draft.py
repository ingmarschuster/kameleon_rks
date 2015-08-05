from kameleon_rks.KameleonRKSGaussian import KameleonRKSGaussian
from kameleon_rks.banana import sample_banana
from kameleon_rks.gaussian_rks import sample_basis, gamma_median_heuristic
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# fix RKS basis
D = 2
m = 500
kernel_gamma = gamma_median_heuristic(sample_banana(N=1000, D=D))
kernel_gamma = 0.5
print "Using kernel_gamma=%.2f" % kernel_gamma
omega, u = sample_basis(D, m, kernel_gamma)


# storing all oracle samples fed to Kameleon
Z_all = []

# Kameleon parameters
eta2 = 300
gamma2 = 0.1

# Schudule for adaptation, toy problem: always update
schedule = lambda t: 1.

# sampler instance
kameleon_rks = KameleonRKSGaussian(D, kernel_gamma, m, gamma2, eta2, schedule)
kameleon_rks.initialise()

# proposals centred at those points
Ys = np.array([[0, 10], [-20, -8], [-20, 9.7], [-10, 0], [0, -3], [10, 0], [20, 9.7]])
colors = ['grey', 'b', 'y', 'r', 'g', 'm', 'black']

# super simple demo: adding more and more oracle samples to "MCMC history"
# and visualising proposals
while True:
    # sample 10 more points and update kameleon every time (cheap rank one update)
    for _ in range(10):
        z_new = sample_banana(N=1, D=D)
        Z_all += [z_new]
        Z = np.vstack(Z_all)
    
        # this is not really an MCMC chain, but rather passing all oracle samples
        # note than Kameleon RKS here only uses the last added sample
        kameleon_rks.update(z_new)
    
    # visualise proposals at some points
    plt.plot(Z[:, 0], Z[:, 1], 'bo')
    for j in range(len(colors)):
        y = Ys[j]

        # draw a number of proposals at the current point
        n_proposals = 50
        X_star = np.zeros((n_proposals, D))
        for i in range(n_proposals):
            
            # sample proposal, not using probability here
            x_star, x_star_log_prob = kameleon_rks.proposal(y)
            X_star[i] = x_star
        
        plt.plot(y[0], y[1], '*', markersize=15, color=colors[j])
        plt.plot(X_star[:, 0], X_star[:, 1], 'x', color=colors[j])
        
    plt.axis('tight')
    plt.xlim([-25, 25])
    plt.ylim([-10, 15])
    plt.ion()
    plt.savefig("kameleon_rks=N=%d.jpg" % len(Z))
    plt.draw()
    plt.show()
    plt.clf()
