''''''
from numpy import array, int32, linspace, pi
from numpy.linalg import inv
from quadrature_rules import jacobi_gauss_lobatto
from polynomials import jacobi_polynomial

def Vandermonde1D(order, quadrature_points):
    '''Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i)'''
    return array([
        jacobi_polynomial(
            quadrature_points,
            0,
            0,
            j - 1)
        for j in range(1, order + 1)])


class Mesh1D:
    def __init__(self, no_of_elements, xmin, xmax):
        self.no_of_elements = no_of_elements
        self.no_of_nodes = no_of_elements + 1
        self.vx = linspace(xmin, xmax, self.no_of_nodes)
        self.edge2node = array(
            [(k, k+1) for k in range(self.no_of_elements)],
            dtype=int32)


if __name__ == '__main__':
    N = 4
    mesh = Mesh1D(N, 0, 2 * pi)

    r = jacobi_gauss_lobatto(N, 0, 0)
    V = Vandermonde1D(N, r)
    invV = inv(V)
    tolerance = 1e-10
