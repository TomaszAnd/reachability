import numpy as np
from scipy.linalg import norm

def compute_Krylov_basis(A, b, m):
    """
    purpose: Comput Krylov basis via Arnoldi iterations
    Compute basis of Krylov space defined as
    span{b,A^1b,A^2b,...,A^(m-1)b} using Arnoldi
    algorithm

    input:
    A : quadratic matrix
    b : column vector
    m : order of

    output:
    V : matrix with columns giving a orthonormal basis of Krylov

    """
    n = A.shape[0]
    result = np.empty((n, m), dtype=np.complex128)  # Use np.complex128 for compatibility with complex types

    # Normalize initial vector
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        raise ValueError("Input vector b must be non-zero.")
    result[:, 0] = b / b_norm

    for index in range(1, m):
        # Multiply the previous basis vector with A
        w = A @ result[:, index - 1]
        # Orthogonalize against previous basis vectors
        h = result[:, :index].conj().T @ w  # complex inner products
        w = w - result[:, :index] @ h

        htilde = np.linalg.norm(w)
        # Handle near-zero vectors
        if htilde < 1e-14:
            # Subspace has degenerated; return the basis so far
            # print(f"Breakdown at iteration {index}")
            result[:, index:] = 0  # fill remaining with zeros if needed
            break
        result[:, index] = w / htilde

    return result


def hamiltonian_generators(D):
    '''
    purpose:
    - compute list of basis vectors of hermitian D x D matrix
    - compute array of flattened/reshaped matrix basis

    input:
    D : dimension of Hermitian matrix

    output:
    basis_mat : matrix made up of columns,
                column is a reshaped basis matrix
    '''
    basis_list = []
    for i in range(D):
        for j in range(i+1, D):
            matrix = np.zeros((D, D), dtype=complex)
            matrix[i, j] = 1.0
            basis_list.append(matrix + matrix.T)
            basis_list.append(1j*matrix - 1j*matrix.T)

    for i in range(D):
        matrix = np.zeros((D, D), dtype=complex)
        matrix[i, j] = 1.0
        basis_list.append(1j*matrix - 1j*matrix.T)

    # reshape matrix basis vectors and write in array
    basis_mat = np.zeros((D**2,D**2)) + 1j*np.zeros((D**2,D**2))
    for i in range(len(basis_list)):
        basis_mat[:,i] = basis_list[i].reshape(D**2)

    return basis_mat
