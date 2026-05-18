import contextlib
import io
import math
import unittest

import ising_ed


class MatrixHelperTests(unittest.TestCase):
    def test_zero_matrix_returns_independent_zero_rows(self):
        matrix = ising_ed.zero_matrix(3)

        self.assertEqual(matrix, [[0.0, 0.0, 0.0] for _ in range(3)])
        matrix[0][0] = 1.0
        self.assertEqual(matrix[1][0], 0.0)
        self.assertEqual(matrix[2][0], 0.0)

    def test_identity_matrix_has_ones_only_on_diagonal(self):
        self.assertEqual(
            ising_ed.identity_matrix(3),
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )


class JacobiEigendecompositionTests(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected, places=10):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=places)

    def test_diagonal_matrix_eigenvalues_are_sorted(self):
        eigenvalues, eigenvectors = ising_ed.jacobi_eigendecomposition(
            [[3.0, 0.0], [0.0, 1.0]]
        )

        self.assertVectorAlmostEqual(eigenvalues, [1.0, 3.0])
        self.assertVectorAlmostEqual([eigenvectors[0][0], eigenvectors[1][0]], [0.0, 1.0])
        self.assertVectorAlmostEqual([eigenvectors[0][1], eigenvectors[1][1]], [1.0, 0.0])

    def test_symmetric_two_by_two_matrix(self):
        eigenvalues, eigenvectors = ising_ed.jacobi_eigendecomposition(
            [[2.0, 1.0], [1.0, 2.0]]
        )

        self.assertVectorAlmostEqual(eigenvalues, [1.0, 3.0])
        for column in range(2):
            norm = math.sqrt(sum(eigenvectors[row][column] ** 2 for row in range(2)))
            self.assertAlmostEqual(norm, 1.0)


class SpinBasisTests(unittest.TestCase):
    def test_spin_bit_reads_sites_from_left_to_right(self):
        # Binary 1010 means sites 0 and 2 are down bits, sites 1 and 3 are up bits.
        self.assertEqual(
            [ising_ed.spin_bit(0b1010, site, 4) for site in range(4)],
            [1, 0, 1, 0],
        )

    def test_flip_site_toggles_requested_left_to_right_site(self):
        self.assertEqual(ising_ed.flip_site(0b1010, 0, 4), 0b0010)
        self.assertEqual(ising_ed.flip_site(0b1010, 1, 4), 0b1110)
        self.assertEqual(ising_ed.flip_site(0b1010, 3, 4), 0b1011)

    def test_basis_label_uses_spin_arrows(self):
        self.assertEqual(ising_ed.basis_label(0b0101, 4), "|↑↓↑↓>")


class HamiltonianTests(unittest.TestCase):
    def test_build_hamiltonian_for_two_sites(self):
        # For L=2, each periodic bond is counted once per site, and h=0 leaves only
        # the diagonal -sum_i z_i z_{i+1} interaction term.
        self.assertEqual(
            ising_ed.build_hamiltonian(length=2, h=0.0),
            [
                [-2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, -2.0],
            ],
        )

    def test_build_hamiltonian_adds_transverse_field_flips(self):
        self.assertEqual(
            ising_ed.build_hamiltonian(length=1, h=0.5),
            [[-1.0, -0.5], [-0.5, -1.0]],
        )

    def test_length_four_ground_state_energy_matches_known_result(self):
        ham = ising_ed.build_hamiltonian(length=4, h=1.0)
        eigenvalues, _ = ising_ed.jacobi_eigendecomposition(ham)

        self.assertAlmostEqual(eigenvalues[0], -5.2262518595, places=10)


class MainTests(unittest.TestCase):
    def test_main_prints_model_summary_and_spin_amplitudes(self):
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            ising_ed.main()

        text = output.getvalue()
        self.assertIn("Model: 1D transverse-field Ising model (periodic), L=4, h=1.0", text)
        self.assertIn("Ground-state energy: -5.2262518595", text)
        self.assertIn("|↑↑↑↑>:", text)
        self.assertIn("|↓↓↓↓>:", text)


if __name__ == "__main__":
    unittest.main()
