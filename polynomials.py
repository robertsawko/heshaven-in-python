from numpy import zeros, sqrt
from scipy.special import gamma


def jacobi_polynomial(x, alpha, beta, order):
    '''Evaluate Jacobi Polynomial of type (alpha,beta) > -1
    (alpha+beta <> -1) at points x for order N and returns P[1:length(xp))]
    Note   : They are normalized to be orthonormal.
    '''
    xp = x.copy()
    PL = zeros((order + 1, xp.size))
    gamma0 = \
        2**(alpha + beta + 1.0) \
        / (alpha + beta + 1.0) \
        * gamma(alpha + 1.0) \
        * gamma(beta + 1.0) \
        / gamma(alpha + beta + 1.0)

    PL[0, :] = 1.0 / sqrt(gamma0)
    if order == 0:
        P = PL.T
        return P
    gamma1 = \
        (alpha + 1) \
        * (beta + 1) \
        / (alpha + beta + 3) \
        * gamma0
    PL[1, :] = \
        (
            (alpha + beta + 2) * xp/2
            + (alpha - beta) / 2
        ) / sqrt(gamma1)
    if order == 1:
        P = PL[order, :].T
        return P
    aold = \
        2.0 / (2.0 + alpha + beta) \
        * sqrt(
            (alpha + 1)
            * (beta + 1)
            / (alpha + beta + 3))

    print(aold)
    for i in range(1, order):
        h1 = 2 * 1 + alpha + beta
        anew = \
            2.0 / (h1 + 2) \
            * sqrt(
                (i + 1)
                * (i + 1 + alpha + beta)
                * (i + 1 + alpha)
                * (i + 1 + beta)
                / (h1 + 1)
                / (h1 + 3))
        bnew = - (alpha**2 - beta**2) / h1 / (h1 + 2)
        print(anew)
        print(bnew)
        print(PL.shape)
        PL[i + 1, :] = \
            1.0 / anew * (
                -aold * PL[i-1:, :]
                + (xp - bnew) * PL[i, :])
        aold = anew
    P = PL[order, :].T
    return P

