import numpy as np
import pytest

from dmrg_from_scratch import (
    compress,
    entropy,
    from_dense,
    left_canonicalize,
    normalize,
    overlap,
    product_mps,
    random_mps,
    right_canonicalize,
    schmidt_values,
    to_dense,
    truncated_svd,
)


def test_overlap_matches_dense():
    a, b = random_mps(7, seed=1), random_mps(7, seed=2)
    assert np.isclose(overlap(a, b), to_dense(a) @ to_dense(b))


def test_canonical_forms_keep_state_and_are_isometries():
    mps = random_mps(8, chi=6, seed=3)
    psi = to_dense(mps)
    left, right = left_canonicalize(mps), right_canonicalize(mps)
    assert np.allclose(to_dense(left), psi)
    assert np.allclose(to_dense(right), psi)
    for A in left[:-1]:
        assert np.allclose(np.einsum("asb,asc->bc", A, A), np.eye(A.shape[2]))
    for B in right[1:]:
        assert np.allclose(np.einsum("asb,csb->ac", B, B), np.eye(B.shape[0]))


def test_product_state_has_no_entanglement():
    mps = product_mps([[1, 0], [0, 1], [1, 1]])
    assert all(np.isclose(entropy(S), 0) for S in schmidt_values(mps))


@pytest.mark.parametrize("cut", [1, 3, 5])
def test_schmidt_values_match_dense_svd(cut):
    mps = random_mps(6, chi=8, seed=4)
    psi = to_dense(normalize(mps))
    S_dense = np.linalg.svd(psi.reshape(2**cut, -1), compute_uv=False)
    S_mps = schmidt_values(mps)[cut - 1]
    assert np.allclose(S_mps, S_dense[: len(S_mps)])


def test_truncated_svd_discarded_weight():
    M = np.random.default_rng(0).normal(size=(10, 10))
    U, S, Vh, disc = truncated_svd(M, chi_max=4)
    s = np.linalg.svd(M, compute_uv=False)
    assert len(S) == 4 and np.isclose(disc, np.sum(s[4:] ** 2) / np.sum(s**2))


def test_compress_error_is_small_for_weakly_entangled_state():
    mps = normalize(random_mps(10, chi=2, seed=5))
    mps[4] = mps[4] + 1e-3 * np.random.default_rng(1).normal(size=mps[4].shape)
    small = compress(mps, chi_max=2)
    fidelity = abs(overlap(small, normalize(mps))) ** 2
    assert fidelity > 0.999


def test_ghz_entropy_is_one_bit():
    ghz = np.zeros(2**5)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    # GHZ as an MPS with bond dimension 2: A[a, s, b] = delta(a, s, b)
    A = np.zeros((2, 2, 2))
    A[0, 0, 0] = A[1, 1, 1] = 1
    mps = [A[0:1] + A[1:2], A, A, A, A[:, :, 0:1] + A[:, :, 1:2]]
    assert np.allclose(to_dense(normalize(mps)), ghz)
    assert all(np.isclose(entropy(S), 1) for S in schmidt_values(mps))  # one bit


def test_from_dense_is_exact_and_left_canonical():
    psi = np.random.default_rng(6).normal(size=2**7)
    psi /= np.linalg.norm(psi)
    mps = from_dense(psi, 7)
    assert np.allclose(to_dense(mps), psi)
    assert max(A.shape[2] for A in mps) == 8  # 2^(L/2) rounded down
