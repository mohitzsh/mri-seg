import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

"""
    Plot the distribution of the displacement parameters
    theta: unnormalized displacement parameters in a tensor of shape 2XHXW
"""
def plot_displacement_distribution(theta,H,W):
        theta = theta.numpy()
        theta_x = (theta[0] + 1)*W
        theta_y = (theta[1] + 1)*H
        theta_norm = np.sqrt(np.square(theta_x) + np.square(theta_y))

        theta_norm = theta_norm.flatten()

        n,_ = np.histogram(theta_norm,bins=range(200))
        y = n/np.sum(n)
        x = np.asarray(list(range(200)))
        x = (x[:-1] + x[1:])/2
        plt.plot(x,y,'.')
        plt.xlabel("Displacement (magnitude)")
        plt.ylabel("Fraction of Pixels")
        plt.title("Distribution of Displacement")
        plt.savefig("test.png")
