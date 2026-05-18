#!/usr/bin/env python3
"""Exact diagonalization for the 1D transverse-field Ising model (L=4, h=1.0)."""

from __future__ import annotations

import math
from typing import List, Tuple

Matrix = List[List[float]]


def zero_matrix(n: int) -> Matrix:
    return [[0.0 for _ in range(n)] for _ in range(n)]


def identity_matrix(n: int) -> Matrix:
    mat = zero_matrix(n)
    for i in range(n):
        mat[i][i] = 1.0
    return mat


def jacobi_eigendecomposition(
    matrix: Matrix, tol: float = 1e-12, max_sweeps: int = 200
) -> Tuple[List[float], Matrix]:
    """Eigen-decomposition of a real symmetric matrix using the Jacobi method."""
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = identity_matrix(n)

    for _ in range(max_sweeps):
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > max_val:
                    max_val = abs(a[i][j])
                    p, q = i, j

        if max_val < tol:
            break

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        if abs(apq) < tol:
            continue

        tau = (aqq - app) / (2.0 * apq)
        t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau)
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        for k in range(n):
            if k != p and k != q:
                akp = a[k][p]
                akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[p][k] = a[k][p]
                a[k][q] = s * akp + c * akq
                a[q][k] = a[k][q]

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(n):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq

    eigenvalues = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigenvalues[i])
    sorted_values = [eigenvalues[i] for i in order]
    sorted_vectors = [[v[row][col] for col in order] for row in range(n)]
    return sorted_values, sorted_vectors


def spin_bit(index: int, site: int, length: int) -> int:
    """Return bit at site (0 for up, 1 for down) with site index from left to right."""
    shift = length - 1 - site
    return (index >> shift) & 1


def flip_site(index: int, site: int, length: int) -> int:
    shift = length - 1 - site
    return index ^ (1 << shift)


def build_hamiltonian(length: int, h: float) -> Matrix:
    """Build H = -Σ σ^z_i σ^z_{i+1} - h Σ σ^x_i (periodic boundary)."""
    dim = 2**length
    ham = zero_matrix(dim)

    for state in range(dim):
        # Diagonal interaction term
        interaction = 0.0
        for i in range(length):
            j = (i + 1) % length
            si = 1.0 if spin_bit(state, i, length) == 0 else -1.0
            sj = 1.0 if spin_bit(state, j, length) == 0 else -1.0
            interaction += si * sj
        ham[state][state] += -interaction

        # Off-diagonal transverse field term
        for i in range(length):
            flipped = flip_site(state, i, length)
            ham[state][flipped] += -h

    return ham


def basis_label(index: int, length: int) -> str:
    bits = format(index, f"0{length}b")
    spins = "".join("↑" if bit == "0" else "↓" for bit in bits)
    return f"|{spins}>"


def main() -> None:
    length = 4
    h = 1.0

    ham = build_hamiltonian(length=length, h=h)
    eigenvalues, eigenvectors = jacobi_eigendecomposition(ham)

    ground_energy = eigenvalues[0]
    ground_state = [eigenvectors[row][0] for row in range(2**length)]

    print(f"Model: 1D transverse-field Ising model (periodic), L={length}, h={h}")
    print(f"Ground-state energy: {ground_energy:.10f}")
    print("Ground-state vector (z-basis ordering |0000>, |0001>, ..., |1111>):")
    print("[" + ", ".join(f"{amp:+.10f}" for amp in ground_state) + "]")

    print("\nAmplitudes by spin configuration:")
    for idx, amp in enumerate(ground_state):
        print(f"{basis_label(idx, length)}: {amp:+.10f}")


if __name__ == "__main__":
    main()
