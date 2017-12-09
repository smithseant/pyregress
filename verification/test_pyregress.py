# -*- coding: utf-8 -*-
"""
Unit testing for the pyregress package.
Created Sep 2017  @author: Sean T. Smith
"""
from unittest import TestCase
from numpy import empty, ones, linspace
from numpy.random import randn
from numba import jit
from pyregress import (GPI, Noise, SquareExp, GammaExp, RatQuad, KernelError,
                       Jeffreys, Uniform)
from DOE_spacefilling import optmaximin

Δ = 1e-4
@jit(nopython=True)
def radius(X):
    Nd, Nx = X.shape
    Rk = empty((Nd, Nd, Nx))
    for i in range(Nd):
        for j in range(Nd):
            for k in range(Nx):
                Rk[i, j, k] = Rk[j, i, k] = X[i, k] - X[j, k]
    return Rk

class PyregressTesting(TestCase):
    def setUp(self):
        Nx = 10   # Number of points in x use to create the kernels
        self.kernels = [
            Noise(w=1.5),
            Noise(w=Jeffreys(1.5)),
            SquareExp(w=Jeffreys(1.5), l=0.8),
            SquareExp(w=1.5, l=Jeffreys(0.8)),
            SquareExp(w=Jeffreys(1.5), l=[0.8, 1.3]),
            SquareExp(w=1.5, l=[Jeffreys(0.8), 1.3]),
            SquareExp(w=1.5, l=[0.8, Jeffreys(1.3)]),
            SquareExp(w=Jeffreys(1.5), l=Jeffreys(0.8)),
            SquareExp(w=Jeffreys(1.5), l=[Jeffreys(0.8), 1.3]),
            SquareExp(w=Jeffreys(1.5), l=[0.8, Jeffreys(1.3)]),
            SquareExp(w=1.5, l=[Jeffreys(0.8), Jeffreys(1.3)]),
            GammaExp(w=Jeffreys(1.5), l=0.8, γ=1.5),
            GammaExp(w=1.5, l=Jeffreys(0.8), γ=1.5),
            GammaExp(w=Jeffreys(1.5), l=[0.8, 1.3], γ=1.5),
            GammaExp(w=1.5, l=[Jeffreys(0.8), 1.3], γ=1.5),
            GammaExp(w=1.5, l=[0.8, Jeffreys(1.3)], γ=1.5),
            GammaExp(w=1.5, l=0.8, γ=Uniform(2, 1.5)),
            GammaExp(w=1.5, l=[0.8, 1.3], γ=Uniform(2, 1.5)),
            GammaExp(w=Jeffreys(1.5), l=Jeffreys(0.8), γ=1.5),
            GammaExp(w=Jeffreys(1.5), l=[Jeffreys(0.8), 1.3], γ=1.5),
            GammaExp(w=Jeffreys(1.5), l=[0.8, Jeffreys(1.3)], γ=1.5),
            GammaExp(w=Jeffreys(1.5), l=0.8, γ=Uniform(2, 1.5)),
            GammaExp(w=Jeffreys(1.5), l=[0.8, 1.3], γ=Uniform(2, 1.5)),
            GammaExp(w=1.5, l=[Jeffreys(0.8), Jeffreys(1.3)], γ=1.5),
            GammaExp(w=1.5, l=Jeffreys(0.8), γ=Uniform(2, 1.5)),
            GammaExp(w=1.5, l=[Jeffreys(0.8), 1.3], γ=Uniform(2, 1.5)),
            GammaExp(w=1.5, l=[0.8, Jeffreys(1.3)], γ=Uniform(2, 1.5)),
            RatQuad(w=Jeffreys(1.5), l=0.8, α=1.5),
            RatQuad(w=1.5, l=Jeffreys(0.8), α=1.5),
            RatQuad(w=Jeffreys(1.5), l=[0.8, 1.3], α=1.5),
            RatQuad(w=1.5, l=[Jeffreys(0.8), 1.3], α=1.5),
            RatQuad(w=1.5, l=[0.8, Jeffreys(1.3)], α=1.5),
            RatQuad(w=1.5, l=0.8, α=Jeffreys(1.5)),
            RatQuad(w=1.5, l=[0.8, 1.3], α=Jeffreys(1.5)),
            RatQuad(w=Jeffreys(1.5), l=Jeffreys(0.8), α=1.5),
            RatQuad(w=Jeffreys(1.5), l=[Jeffreys(0.8), 1.3], α=1.5),
            RatQuad(w=Jeffreys(1.5), l=[0.8, Jeffreys(1.3)], α=1.5),
            RatQuad(w=Jeffreys(1.5), l=0.8, α=Jeffreys(1.5)),
            RatQuad(w=Jeffreys(1.5), l=[0.8, 1.3], α=Jeffreys(1.5)),
            RatQuad(w=1.5, l=[Jeffreys(0.8), Jeffreys(1.3)], α=1.5),
            RatQuad(w=1.5, l=Jeffreys(0.8), α=Jeffreys(1.5)),
            RatQuad(w=1.5, l=[Jeffreys(0.8), 1.3], α=Jeffreys(1.5)),
            RatQuad(w=1.5, l=[0.8, Jeffreys(1.3)], α=Jeffreys(1.5)),
            Noise(w=Jeffreys(1.5)) +
               SquareExp(w=1.5, l=[0.8, Jeffreys(1.3)])]#,
            # SquareExp(w=1.5, l=Jeffreys(0.8)) *
            #    SquareExp(w=1.5, l=Jeffreys(1.3))]
        self.Nk = len(self.kernels)
        self.Nφ = [0] + [1]*6 + [2]*4 + [1]*7 + [2]*9 + [1]*7 + [2]*11
        self.Xd = [None] * self.Nk
        self.Rk = [None] * self.Nk
        self.Ys = [None] * self.Nk
        for ik, kern in zip(range(self.Nk), self.kernels):
            if 'l' not in kern.p:
                if (isinstance(kern, Noise) or
                    not isinstance(kern.terms[-1].p['l'], list)):
                    Nd = 1
                    self.Xd[ik] = ones((Nx, Nd))
                else:
                    Nd = len(kern.terms[-1].p['l'])
                    design = optmaximin(Nx**Nd, Nd, method='lhd',
                                        n_samples=100, verbose=False)
                    self.Xd[ik] = 5 * design.x
            else:
                if not isinstance(kern.p['l'], list):
                    Nd = 1
                    self.Xd[ik] = linspace(0, 5, Nx).reshape([Nx, Nd])
                else:
                    Nd = len(kern.p['l'])
                    design = optmaximin(Nx**Nd, Nd, method='lhd',
                                        n_samples=100, verbose=False)
                    self.Xd[ik] = 5 * design.x
            self.Rk[ik] = radius(self.Xd[ik])
            Xe = empty((0, Nd))
            Ye = empty(0)
            eGPI = GPI(Xe, Ye, kern, optimize=False)
            self.Ys[ik] = eGPI.sample(self.Xd[ik])

    def test_kernel_Nφ(self):
        """
        Ensure kernel has identified the expected number of hyper-parameters.
        """
        for ik in range(self.Nk):
            self.assertEqual(self.Nφ[ik], self.kernels[ik].Nφ,
                             msg='Kernel No. {}'.format(ik))

    def test_kernel_Kφ(self):
        """
        Ensure the kernel __call__ method and Kφ return the same base values.
        """
        tol = 1e-12
        for ik, kern, Rk in zip(range(self.Nk), self.kernels, self.Rk):
            φ = kern.get_φ(trans=False)
            Kx = kern(Rk)
            Kφ = kern.Kφ(φ, Rk)
            rel_err = abs((Kx - Kφ) / (Kx + 1e-8))
            self.assertLess(rel_err.max(), tol)

    def test_kernel_gradx(self):
        """
        Ensure the kernel is returning function values & spatial (radial)
        gradients that are consistent (within a numerical tolerance).
        """
        tol = 1e-5  # Absolute tolerance on the gradient error
        for ik, kern in zip(range(self.Nk), self.kernels):
            try:
                K, gradK = kern(self.Rk[ik], grad=True, trans=False)
            except KernelError:
                continue
            Nd = self.Xd[ik].shape[1]
            diffK = empty(gradK.shape)
            for ix in range(Nd):
                R = self.Rk[ik].copy()
                R[:, :, ix] = self.Rk[ik][:, :, ix] - Δ
                Kneg = kern(R)
                R[:, :, ix] = self.Rk[ik][:, :, ix] + Δ
                Kpos = kern(R)
                diffK[:, :, ix] = (Kpos - Kneg) / (2 * Δ)
            rel_err = abs((gradK - diffK) / (gradK + 1e-8))
            self.assertLess(rel_err.max(), tol,
                            msg=('Kernel No. {}').format(ik))

    def test_kernel_gradφ(self):
        """
        Ensure the kernel is returning function values & hyper-parameter
        gradients that are consistent (within a numerical tolerance).
        """
        tol = 5e-3  # tolerance on the gradient error
        Ns = 10  # Number of points in φ where the test is performed
        for ik, kern in zip(range(self.Nk), self.kernels):
            if kern.Nφ == 0:
                continue
            φ0 = kern.get_φ(trans=False)
            Φ = empty((Ns, kern.Nφ))
            for iφ, φdist in zip(range(kern.Nφ), kern.iter_φdist()):
                Φ[:, iφ] = φdist.invtr(randn(Ns) / 4 + φdist.transformed.guess)
            for j in range(Ns):
                φ = Φ[j].copy()
                _, gradK = kern.Kφ(φ, self.Rk[ik], grad=True, trans=False)
                for iφ in range(kern.Nφ):
                    φ[iφ] = Φ[j, iφ] - Δ
                    kern.update_p(φ, trans=False, set=True)
                    Kneg = kern.Kφ(φ, self.Rk[ik])
                    φ[iφ] = Φ[j, iφ] + Δ
                    kern.update_p(φ, trans=False, set=True)
                    Kpos = kern.Kφ(φ, self.Rk[ik])
                    diffK = (Kpos - Kneg) / (2 * Δ)
                    rel_err = abs((gradK[:, :, iφ] - diffK) /
                                  (gradK[:, :, iφ] + 1e-8))
                    self.assertLess(rel_err.max(), tol,
                                msg=('Kernel No. {}, hyper-parameter No. {},'+
                                     'spatial point No. {}').format(ik, iφ, j))
                    φ[iφ] = Φ[j, iφ]
            kern.update_p(φ0, trans=False, set=True)

    def test_interpolant_consistency(self):
        """
        Ensure that an interpolant can reproduce its source data.
        """
        tol = 1e-7
        for ik, kern in zip(range(self.Nk), self.kernels):
            if (isinstance(kern, Noise) or hasattr(kern, 'terms') and
                any([isinstance(t, Noise) for t in kern.terms])):
                continue
            GPIk = GPI(self.Xd[ik], self.Ys[ik], kern, optimize=False)
            Yd = GPIk(self.Xd[ik])
            rel_err = abs((self.Ys[ik] - Yd) / (self.Ys[ik] + 1e-8))
            self.assertLess(rel_err.max(), tol, msg='Kernel No. {}'.format(ik))

    # TODO: The test to reproduce φ may not be well posed - fix.
    # def test_convergence_of_maximize_posterior_φ(self):
    #     """
    #     Ensure the optimization can reproduce the φ that was used in the
    #     random generation of its source data.
    #     Stated chronologically:  Start with a kernel that has specified
    #     parameters. Next, sample function values from the prior (GPI with
    #     empty data arrays). Then, make a new inference that uses
    #     those function values as data and uses the same kernel type but
    #     unknown φ values. Finally, check that the optimization converges to
    #     the φ values used in the initial sampling.
    #     """
    #     tol = 1e-2
    #     for ik, kern in zip(range(self.Nk), self.kernels):
    #         if kern.Nφ == 0 or isinstance(kern, Noise):
    #             continue
    #         φ_old = kern.get_φ(trans=False).copy()
    #         from numpy.linalg import LinAlgError
    #         try:
    #             GPIk = GPI(self.Xd[ik], self.Ys[ik], kern,
    #                        optimize=True)
    #         except LinAlgError as e:
    #             from pickle import dump
    #             with open('failing_kernel_1.p', 'wb') as file:
    #                 dump((ik, self.Xd[ik], self.Ys[ik]), file)
    #             raise e
    #         φ_new = kern.get_φ(trans=False)
    #         rel_err = abs((φ_old - φ_new) / (φ_old + 1e-8))
    #         print(ik, rel_err.max())
    #         # self.assertLess(rel_err.max(), tol, msg='Kernel No. {}'.format(ik))
    #         kern.update_p(φ_old, trans=False, set=True)

    # TODO: Test if the explicit bases reproduce the expected polynomials.
    # TODO: Test whether the transformations behave as expected.

if __name__ == '__main__':
    from unittest import main
    main()