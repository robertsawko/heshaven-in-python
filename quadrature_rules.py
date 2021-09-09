'''
Code related to quadrature rules
'''
from numpy import arange, array, concatenate, diag, finfo, sqrt, zeros
from numpy.linalg import eig
from scipy.special import gamma


def jacobi_gauss_quadrature(order, alpha, beta):
    ''' Compute the N'th order Gauss quadrature points, x, and weights, w,
    associated with the Jacobi polynomial, of type
    (alpha,beta) > -1 ( <> -0.5).'''
    if order == 0:
        x = -(alpha - beta) / (alpha + beta + 2.0)
        w = 2.0
        return array([x]), array([w])
    h1 = 2 * arange(order + 1) + alpha + beta
    h2 = arange(1.0, order + 1)
    J = \
        diag(-0.5 * (alpha**2 - beta**2) / (h1 + 2.0) / h1) \
        + diag(
            2.0 / (h1[:-1] + 2.0)
            * sqrt(
                h2
                * (h2 + alpha + beta)
                * (h2 + alpha)
                * (h2 + beta)
                / (h1[:-1] + 1.0)
                / (h1[:-1] + 3.0)),
            1)
    if alpha + beta < 10 * finfo(float).eps:
        J[0, 0] = 0.0
    J = J + J.T
    x, V = eig(J)
    w = \
        (V[0, :].T)**2 \
        * 2**(alpha + beta + 1.0) \
        / (alpha + beta + 1.0) \
        * gamma(alpha + 1.0) \
        * gamma(beta + 1.0) \
        / gamma(alpha + beta + 1.0)
    return x.T, w


def jacobi_gauss_lobatto(order, alpha, beta):
    ''' Compute the N'th order Gauss Lobatto quadrature points, x, associated
    with the Jacobi polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    '''
    x = zeros(order + 1)

    if order == 1:
        x[0] = -1.0
        x[1] = 1.0
    else:
        x_internal, _ = jacobi_gauss_quadrature(
            order - 2,
            alpha + 1.0,
            beta + 1.0,)
        x = concatenate(([-1.0], x_internal, [1.0]))
    return x
