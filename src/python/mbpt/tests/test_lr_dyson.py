"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================

Test suite for Linear Response Dyson equation.

Tests include:
1. k+q mapping utilities (C++ binding)
2. LR Dyson equation validation against finite difference

NOTE: TestLRDyson validates the pure-numpy reference implementation defined in
this file against numpy finite differences; it does NOT exercise the C++
lr_dyson solver. C++ coverage lives in src/methods/SCF/tests/test_lr_dyson.cpp
and the lr_debug / lr_tests suites (via the run_lr Python API).
"""

import numpy as np
import pytest
import tempfile
import os
from typing import Tuple

from coqui.mbpt.lr_driver import (
    calculate_kpq_map,
    is_q_commensurate,
    is_q_gamma,
    write_DeltaH0,
    read_DeltaH0,
)


# =============================================================================
# Pure Python/NumPy reference implementations for validation
# These are test-only implementations, kept here per project guidelines.
# =============================================================================

def solve_lr_dyson_fixed_sigma_numpy(
    G_wskij: np.ndarray,
    DeltaH0_skij: np.ndarray,
    kpq_map: np.ndarray
) -> np.ndarray:
    """
    Solve LR Dyson equation with fixed self-energy (pure Python/NumPy).

    Computes: ΔG(k,iω) = G(k+q,iω) · ΔH0(k) · G(k,iω)
    """
    nw, ns, nk, nb, _ = G_wskij.shape
    DeltaG_wskij = np.zeros_like(G_wskij)

    for iw in range(nw):
        for is_ in range(ns):
            for ik in range(nk):
                ikq = kpq_map[ik]
                G_k = G_wskij[iw, is_, ik]
                G_kq = G_wskij[iw, is_, ikq]
                DeltaH0_k = DeltaH0_skij[is_, ik]
                DeltaG_wskij[iw, is_, ik] = G_kq @ DeltaH0_k @ G_k

    return DeltaG_wskij


def solve_dyson_numpy(
    H0_skij: np.ndarray,
    F_skij: np.ndarray,
    Sigma_wskij: np.ndarray,
    S_skij: np.ndarray,
    mu: float,
    iwn: np.ndarray
) -> np.ndarray:
    """
    Solve standard Dyson equation (pure Python/NumPy).

    Computes: G(k,iω) = [iω·S - H0 - F - Σ(iω)]^{-1}
    """
    nw, ns, nk, nb, _ = Sigma_wskij.shape
    G_wskij = np.zeros_like(Sigma_wskij)

    for iw in range(nw):
        omega_mu = 1j * iwn[iw] + mu
        for is_ in range(ns):
            for ik in range(nk):
                X = omega_mu * S_skij[is_, ik] - H0_skij[is_, ik] - \
                    F_skij[is_, ik] - Sigma_wskij[iw, is_, ik]
                G_wskij[iw, is_, ik] = np.linalg.inv(X)

    return G_wskij


def validate_lr_dyson_finite_difference(
    H0_skij: np.ndarray,
    F_skij: np.ndarray,
    Sigma_wskij: np.ndarray,
    S_skij: np.ndarray,
    mu: float,
    iwn: np.ndarray,
    DeltaH0_skij: np.ndarray,
    kpq_map: np.ndarray,
    eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Validate LR Dyson against finite difference.
    """
    G_wskij = solve_dyson_numpy(H0_skij, F_skij, Sigma_wskij, S_skij, mu, iwn)
    DeltaG_lr = solve_lr_dyson_fixed_sigma_numpy(G_wskij, DeltaH0_skij, kpq_map)
    G_plus = solve_dyson_numpy(H0_skij + eps * DeltaH0_skij, F_skij,
                                Sigma_wskij, S_skij, mu, iwn)
    G_minus = solve_dyson_numpy(H0_skij - eps * DeltaH0_skij, F_skij,
                                 Sigma_wskij, S_skij, mu, iwn)
    DeltaG_fd = (G_plus - G_minus) / (2 * eps)
    max_error = np.max(np.abs(DeltaG_lr - DeltaG_fd))
    return DeltaG_lr, DeltaG_fd, max_error


class TestKpqMapping:
    """Tests for k+q mapping utilities."""

    def test_calculate_kpq_map_gamma(self):
        """Test k+q mapping for q=0 (identity mapping)."""
        # 2x2x2 k-point grid
        kpts = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ])
        q_vec = np.array([0.0, 0.0, 0.0])

        kpq_map = calculate_kpq_map(kpts, q_vec)

        # For q=0, should be identity
        np.testing.assert_array_equal(kpq_map, np.arange(8))

    def test_calculate_kpq_map_nonzero_q(self):
        """Test k+q mapping for non-zero q."""
        # 2x2x2 k-point grid
        kpts = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ])
        q_vec = np.array([0.5, 0.0, 0.0])

        kpq_map = calculate_kpq_map(kpts, q_vec)

        # Check that each k+q maps correctly
        for ik in range(len(kpts)):
            kpq = kpts[ik] + q_vec
            kpq = kpq - np.floor(kpq)  # Periodic boundary
            ikq = kpq_map[ik]
            diff = kpts[ikq] - kpq
            diff = diff - np.round(diff)
            assert np.sum(diff**2) < 1e-10

    def test_is_q_commensurate_true(self):
        """Test is_q_commensurate returns True for commensurate q."""
        kpts = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ])
        q_vec = np.array([0.5, 0.0, 0.0])

        assert is_q_commensurate(kpts, q_vec) is True

    def test_is_q_commensurate_false(self):
        """Test is_q_commensurate returns False for incommensurate q."""
        kpts = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ])
        q_vec = np.array([0.25, 0.0, 0.0])  # Not on grid

        assert is_q_commensurate(kpts, q_vec) is False

    def test_is_q_gamma(self):
        """Test is_q_gamma detection."""
        assert is_q_gamma(np.array([0.0, 0.0, 0.0])) is True
        assert is_q_gamma(np.array([1.0, 0.0, 0.0])) is True  # Equivalent to 0
        assert is_q_gamma(np.array([0.5, 0.0, 0.0])) is False
        assert is_q_gamma(np.array([1e-8, 0.0, 0.0])) is True


class TestLRDyson:
    """Tests for LR Dyson equation solver."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple 2-band, 2-kpoint system for testing."""
        nw, ns, nk, nb = 4, 1, 2, 2

        # Random Hermitian H0
        np.random.seed(42)
        H0_skij = np.zeros((ns, nk, nb, nb), dtype=complex)
        for is_ in range(ns):
            for ik in range(nk):
                h = np.random.randn(nb, nb) + 1j * np.random.randn(nb, nb)
                H0_skij[is_, ik] = (h + h.conj().T) / 2

        # Identity overlap
        S_skij = np.zeros((ns, nk, nb, nb), dtype=complex)
        for is_ in range(ns):
            for ik in range(nk):
                S_skij[is_, ik] = np.eye(nb)

        # Small self-energy (Hartree-like)
        Sigma_wskij = np.zeros((nw, ns, nk, nb, nb), dtype=complex)
        for iw in range(nw):
            for is_ in range(ns):
                for ik in range(nk):
                    sigma = np.random.randn(nb, nb) * 0.1
                    Sigma_wskij[iw, is_, ik] = (sigma + sigma.conj().T) / 2

        # Zero Fock matrix for simplicity
        F_skij = np.zeros((ns, nk, nb, nb), dtype=complex)

        # Matsubara frequencies
        beta = 10.0
        iwn = np.array([(2*n + 1) * np.pi / beta for n in range(nw)])

        mu = 0.0

        return {
            'nw': nw, 'ns': ns, 'nk': nk, 'nb': nb,
            'H0_skij': H0_skij,
            'S_skij': S_skij,
            'Sigma_wskij': Sigma_wskij,
            'F_skij': F_skij,
            'iwn': iwn,
            'mu': mu,
            'beta': beta,
        }

    def test_solve_dyson_numpy(self, simple_system):
        """Test that solve_dyson_numpy produces correct Green's function."""
        G = solve_dyson_numpy(
            simple_system['H0_skij'],
            simple_system['F_skij'],
            simple_system['Sigma_wskij'],
            simple_system['S_skij'],
            simple_system['mu'],
            simple_system['iwn']
        )

        # Check shape
        assert G.shape == simple_system['Sigma_wskij'].shape

        # Check that G^{-1} = iw*S - H0 - F - Sigma
        for iw in range(simple_system['nw']):
            omega_mu = 1j * simple_system['iwn'][iw] + simple_system['mu']
            for is_ in range(simple_system['ns']):
                for ik in range(simple_system['nk']):
                    G_inv = np.linalg.inv(G[iw, is_, ik])
                    expected = (omega_mu * simple_system['S_skij'][is_, ik] -
                                simple_system['H0_skij'][is_, ik] -
                                simple_system['F_skij'][is_, ik] -
                                simple_system['Sigma_wskij'][iw, is_, ik])
                    np.testing.assert_allclose(G_inv, expected, rtol=1e-10)

    def test_lr_dyson_finite_difference_q0(self, simple_system):
        """Validate LR Dyson against finite difference for q=0."""
        # Create a random perturbation
        np.random.seed(123)
        ns, nk, nb = simple_system['ns'], simple_system['nk'], simple_system['nb']
        DeltaH0 = np.random.randn(ns, nk, nb, nb) + 1j * np.random.randn(ns, nk, nb, nb)
        DeltaH0 = (DeltaH0 + DeltaH0.conj().transpose(0, 1, 3, 2)) / 2  # Hermitian

        # k+q mapping for q=0 is identity
        kpq_map = np.arange(nk)

        DeltaG_lr, DeltaG_fd, max_error = validate_lr_dyson_finite_difference(
            simple_system['H0_skij'],
            simple_system['F_skij'],
            simple_system['Sigma_wskij'],
            simple_system['S_skij'],
            simple_system['mu'],
            simple_system['iwn'],
            DeltaH0,
            kpq_map,
            eps=1e-6
        )

        # Error should be O(eps^2) for central difference
        assert max_error < 1e-5, f"Max error {max_error} exceeds tolerance"

    def test_lr_dyson_scaling(self, simple_system):
        """Test that LR Dyson scales linearly with perturbation amplitude."""
        ns, nk, nb = simple_system['ns'], simple_system['nk'], simple_system['nb']
        np.random.seed(456)
        DeltaH0 = np.random.randn(ns, nk, nb, nb) + 1j * np.random.randn(ns, nk, nb, nb)
        DeltaH0 = (DeltaH0 + DeltaH0.conj().transpose(0, 1, 3, 2)) / 2

        kpq_map = np.arange(nk)

        G = solve_dyson_numpy(
            simple_system['H0_skij'],
            simple_system['F_skij'],
            simple_system['Sigma_wskij'],
            simple_system['S_skij'],
            simple_system['mu'],
            simple_system['iwn']
        )

        # Compute ΔG for two different amplitudes
        DeltaG_1 = solve_lr_dyson_fixed_sigma_numpy(G, DeltaH0, kpq_map)
        DeltaG_2 = solve_lr_dyson_fixed_sigma_numpy(G, 2 * DeltaH0, kpq_map)

        # Should scale linearly
        np.testing.assert_allclose(DeltaG_2, 2 * DeltaG_1, rtol=1e-12)


class TestH5IO:
    """Tests for HDF5 I/O of linear response data."""

    def test_write_read_DeltaH0(self):
        """Test writing and reading DeltaH0 from HDF5."""
        ns, nk, nb = 1, 4, 3
        np.random.seed(789)
        q_vec = np.array([0.5, 0.0, 0.0])
        DeltaH0 = np.random.randn(ns, nk, nb, nb) + 1j * np.random.randn(ns, nk, nb, nb)

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_lr.h5")

            write_DeltaH0(filename, q_vec, DeltaH0)
            q_vec_read, DeltaH0_read = read_DeltaH0(filename)

            np.testing.assert_allclose(q_vec_read, q_vec)
            np.testing.assert_allclose(DeltaH0_read, DeltaH0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
