# -*- coding: utf-8 -*-
"""
Created Nov 2017 @author: Sean T. Smith
"""
__all__ = ['PiecewiseLinear']

from numpy import empty, ones
from numpy.linalg import solve
from scipy.spatial import Delaunay
from .gaussian_processes.gaussian_process import radius

class PiecewiseLinear:
    """
    Create a class for piecewise-linear interpolation tool that uses Delaunay
    triangulation. This method has limitations that a user should be aware of.
    The largest concern is the regions for that are external to any simplices
    (extrapolation). It extrapolate, but exhibits discontinuities.
    The second largest concern is the behavior of long-skinny simplices.
    """
    def __init__(self, Xd, Yd):
        self.tri = Delaunay(Xd)  # triangulation
        self.ntri = self.tri.nsimplex  # number of simplices
        self.ndim = self.tri.ndim  # number of dimensions
        # simplex centers:
        self.r = empty((self.ntri, self.ndim))
        for i in range(self.ntri):
            tot = 0
            for j in range(self.ndim + 1):
                tot += self.tri.points[self.tri.simplices[i, j]]
            self.r[i, :] = tot / (self.ndim + 1)
        # simplex linear coefficients
        self.θ = empty((self.ntri, self.ndim + 1))
        Ca = empty((self.ndim + 1, self.ndim + 1))
        Ca[:, 0] = 1
        for i in range(self.ntri):
            X = self.tri.points[self.tri.simplices[i]]
            Ca[:, 1:] = X - self.r[i]
            self.θ[i] = solve(Ca, Yd[self.tri.simplices[i]]).reshape(-1)
    def __call__(self, Xi):
        """Perform the interpolation/extrapolation."""
        ni = Xi.shape[0]
        Yi = empty(ni)
        for i in range(ni):
            s = self.tri.find_simplex(Xi[i])
            if s < 0:  # Extrapolate:
                r = radius(Xi[i].reshape((1, -1)), self.r, ones(self.ndim))
                R = (r**2).sum(axis=2).reshape(-1)
                s = R.argmin()
            c = Xi[i] - self.r[s]
            Yi[i] = self.θ[s, 0] + c @ self.θ[s, 1:]
        return Yi


if __name__ == '__main__':
    from numpy import array, linspace, meshgrid
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from src import GPI, SquareExp
    from DOE_spacefilling import Design, optmaximin, potentialfield

    n_points = 11
    lhd = Design(n_points, 2, 'lhd', position='edges')
    # lhd = optmaximin(n_points, 2, n_samples=2000, verbose=False,
    #                  position='edges').x
    lhd = potentialfield(lhd, verbose=False).x
    myGPI = GPI(empty((0, 2)), empty(0), SquareExp(w=10, l=[0.4, 0.8]))
    Y = myGPI.sample(lhd)

    myPL = PiecewiseLinear(lhd, Y)

    ni = 100
    xi1 = linspace(0, 1, ni)
    xi2 = linspace(0, 1, ni + 1)
    Xi1, Xi2 = meshgrid(xi1, xi2, indexing='ij')

    Xi = array([Xi1.reshape(-1), Xi2.reshape(-1)]).T
    Yi = myPL(Xi)
    Yi = Yi.reshape((ni, ni+1))

    # Plot the results:
    fig = plt.figure(figsize=(14, 6))
    fig.add_subplot(1, 2, 1)
    plt.plot(lhd[:, 0], lhd[:, 1], 'o', label='source points')
    plt.triplot(lhd[:, 0], lhd[:, 1], myPL.tri.simplices.copy(), linewidth=2.0,
                label='triangulation')
    plt.plot(myPL.r[:, 0], myPL.r[:, 1], 'o', label='barycentric centers')
    plt.title('Triangulation')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.plot_surface(Xi1, Xi2, Yi.T, rstride=1, cstride=1)
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(Xi1, Xi2, Yi, 40, linewidths=2)
    plt.triplot(lhd[:, 0], lhd[:, 1], myPL.tri.simplices.copy(), linewidth=0.5,
                color=[0.8, 0.8, 0.8], label='triangulation')
    plt.title('Piecewise Linear Interpolation')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.show()
