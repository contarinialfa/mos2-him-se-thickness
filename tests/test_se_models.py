"""Acceptance tests for scripts/fit_se_models.py.

Numerical acceptance tests for the SE-escape model implementation,
run against the in-repo SRIM output. Runnable directly::

    python tests/test_se_models.py

or via pytest. A nonzero exit code means at least one test failed.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fit_se_models as fsm  # noqa: E402


def make_deposition(dz: float = 0.25) -> fsm.DepositionModel:
    srim_z, srim_d = fsm.load_ioniz(fsm.DEFAULT_IONIZ)
    return fsm.DepositionModel(srim_z, srim_d, dz=dz)


def total_energy(dep: fsm.DepositionModel, profile: np.ndarray) -> float:
    return float(np.trapezoid(profile, dep.z))


def test_parser_round_trip() -> None:
    """SRIM parser: bin count, units, z=0 flat extension, monotone grid."""
    srim_z, srim_d = fsm.load_ioniz(fsm.DEFAULT_IONIZ)
    assert len(srim_z) == 101, f"expected 100 bins + z=0, got {len(srim_z)}"
    assert srim_z[0] == 0.0 and srim_d[0] == srim_d[1], "flat z=0 extension"
    # First data row: 300100.E-04 Ang = 3.001 nm; 1483.68E-02 eV/Ang.
    assert np.isclose(srim_z[1], 3.001), srim_z[1]
    assert np.isclose(srim_d[1], 148.368), srim_d[1]
    assert np.isclose(srim_z[-1], 300.001), srim_z[-1]
    assert np.all(np.diff(srim_z) > 0.0), "depth grid must increase"


def test_energy_conservation() -> None:
    """Coordinate remapping conserves integrated energy to < 0.1%.

    This is an identity of the model construction, so failure indicates an
    implementation bug rather than a tolerance issue.
    """
    dep = make_deposition()
    reference = total_energy(dep, dep.base)
    for rho, lam_d in [(0.3, 150.0), (0.8, 50.0), (0.05, 400.0), (0.9, 20.0)]:
        energy = total_energy(dep, dep.profile_m2(rho, lam_d))
        assert abs(energy / reference - 1.0) < 1e-3, (
            f"M2 rho={rho} lam_d={lam_d}: {energy / reference - 1.0:.2e}"
        )
    # alpha >= 0.5 keeps the stretched tail inside the 800 nm grid.
    for chi0, alpha in [(0.3, 0.7), (0.7, 0.5), (0.9, 0.9)]:
        energy = total_energy(dep, dep.profile_m3(chi0, alpha))
        assert abs(energy / reference - 1.0) < 1e-3, (
            f"M3 chi0={chi0} alpha={alpha}: {energy / reference - 1.0:.2e}"
        )
    for chi0, alpha, lam_d in [
        (0.3, 0.7, 150.0),
        (0.6, 0.5, 50.0),
        (0.2, 0.9, 300.0),
    ]:
        energy = total_energy(dep, dep.profile_full(chi0, alpha, lam_d))
        assert abs(energy / reference - 1.0) < 1e-3, (
            f"full chi0={chi0} alpha={alpha} lam_d={lam_d}: "
            f"{energy / reference - 1.0:.2e}"
        )


def test_full_model_limits() -> None:
    """chi0->0, alpha->1, lam_d->0 reduce to the baseline; lam_d->inf to M3."""
    dep = make_deposition()
    scale = float(np.max(dep.base))

    exact = dep.profile_full(0.0, 0.7, 150.0)
    assert np.max(np.abs(exact - dep.base)) / scale < 1e-12, "chi0 -> 0"

    near_one = dep.profile_full(0.5, 1.0, 150.0)
    assert np.max(np.abs(near_one - dep.base)) / scale < 1e-3, "alpha -> 1"

    # lam_d -> 0: ions dechannel immediately, so the profile reduces to the
    # baseline OUTSIDE a boundary layer of width O(lam_d) -- at z = 0 the
    # still-channeled term contributes chi0*(1-alpha)*I_S(0) for any lam_d,
    # so the comparison must exclude z < a few dechanneling lengths. A finer
    # quadrature resolves the lam_d-wide dechanneling spike.
    lam_d_small = 0.5
    fast_dechannel = dep.profile_full(0.5, 0.7, lam_d_small, n_quad=2000)
    outside_layer = dep.z >= 20.0 * lam_d_small
    assert (
        np.max(np.abs((fast_dechannel - dep.base)[outside_layer])) / scale
        < 1e-2
    ), "lam_d -> 0 outside the boundary layer"
    energy_gap = np.trapezoid(np.abs(fast_dechannel - dep.base), dep.z)
    assert energy_gap / np.trapezoid(dep.base, dep.z) < 1e-2, (
        "lam_d -> 0 in the energy (L1) sense"
    )

    slow_dechannel = dep.profile_full(0.5, 0.7, 1e5)
    m3 = dep.profile_m3(0.5, 0.7)
    assert np.max(np.abs(slow_dechannel - m3)) / scale < 5e-3, "lam_d -> inf"


def test_m2_matches_full_model_to_first_order() -> None:
    """M2 tracks the full model to first order in chi0*(1-alpha)."""
    dep = make_deposition()
    scale = float(np.max(dep.base))
    lam_d = 150.0

    deviations = {}
    for alpha in (0.95, 0.90):
        chi0 = 0.4
        full = dep.profile_full(chi0, alpha, lam_d)
        shift = dep.profile_m2(chi0 * (1.0 - alpha), lam_d)
        baseline_dev = np.max(np.abs(full - dep.base)) / scale
        m2_dev = np.max(np.abs(full - shift)) / scale
        deviations[alpha] = (m2_dev, baseline_dev)
        assert m2_dev < 0.5 * baseline_dev, (
            f"alpha={alpha}: M2 ({m2_dev:.3e}) should beat the baseline "
            f"({baseline_dev:.3e})"
        )
    small = deviations[0.95][0]
    large = deviations[0.90][0]
    assert small < 0.01, f"near-first-order deviation too large: {small:.3e}"
    # First-order accuracy: the residual shrinks faster than epsilon itself.
    ratio = large / small
    assert 1.5 < ratio < 10.0, (
        f"expected roughly quadratic residual scaling, got ratio {ratio:.2f}"
    )
    print(
        "  m2-vs-full deviations (rel. to max D): "
        f"eps=0.02: {small:.3e}, eps=0.04: {large:.3e}, ratio {ratio:.2f}"
    )


def test_back_escape_integral() -> None:
    """Y_b matches brute-force quadrature and stays finite at extreme u/lam."""
    dep = make_deposition()
    lam = 5.0
    for u in (17.3, 99.9, 300.0, 310.0):
        # Reference: brute-force quadrature of the SAME uniform-grid
        # discretization the integrator consumes (its documented contract).
        z_fine = np.linspace(0.0, u, 200_001)
        d_fine = np.interp(z_fine, dep.z, dep.base)
        brute = np.trapezoid(d_fine * np.exp(-(u - z_fine) / lam), z_fine)
        front_brute = np.trapezoid(d_fine * np.exp(-z_fine / lam), z_fine)
        y_front, y_back = fsm.escape_integrals(
            dep.z, dep.dz, dep.base, lam, np.array([u])
        )
        assert abs(y_back[0] / brute - 1.0) < 1e-4, (
            f"u={u}: {y_back[0]:.6f} vs brute {brute:.6f}"
        )
        assert abs(y_front[0] / front_brute - 1.0) < 1e-4, f"Y_f at u={u}"
    # Overflow regime: u/lam > 1300 must not produce inf/nan.
    y_front, y_back = fsm.escape_integrals(
        dep.z, dep.dz, dep.base, 0.3, np.array([400.0, 799.9])
    )
    assert np.all(np.isfinite(y_front)) and np.all(np.isfinite(y_back))
    assert np.all(y_back >= 0.0)
    # Far past the deposition range, Y_b decays to (numerically) nothing.
    _, y_far = fsm.escape_integrals(
        dep.z, dep.dz, dep.base, 5.0, np.array([555.5])
    )
    assert np.isfinite(y_far[0]) and 0.0 <= y_far[0] < 1e-15, y_far[0]


def test_continuity_in_u() -> None:
    """Escape integrals are Lipschitz-continuous across grid nodes."""
    dep = make_deposition()
    rng = np.random.default_rng(7)
    epsilon = 1e-7
    profiles = {
        "base": dep.base,
        "m2": dep.profile_m2(0.3, 150.0),
        "m3": dep.profile_m3(0.4, 0.7),
    }
    for name, profile in profiles.items():
        for lam in (0.7, 7.0, 55.0):
            u_random = rng.uniform(0.0, 400.0, 200)
            u_nodes = dep.z[rng.integers(1, len(dep.z) - 1, 100)]
            u_all = np.concatenate([u_random, u_nodes])
            f_hi, b_hi = fsm.escape_integrals(
                dep.z, dep.dz, profile, lam, u_all + epsilon
            )
            f_lo, b_lo = fsm.escape_integrals(
                dep.z, dep.dz, profile, lam, u_all - epsilon
            )
            d_max = float(np.max(profile))
            b_bound = (d_max + np.max(b_hi) / lam) * 2.0 * epsilon * 5.0
            f_bound = d_max * 2.0 * epsilon * 5.0
            assert np.max(np.abs(f_hi - f_lo)) < f_bound, (
                f"Y_f jump ({name}, lam={lam})"
            )
            assert np.max(np.abs(b_hi - b_lo)) < b_bound, (
                f"Y_b jump ({name}, lam={lam})"
            )


def test_variable_projection() -> None:
    """The profiled amplitude is the exact conditional optimum."""
    dep = make_deposition()
    data = fsm.load_experimental(fsm.DEFAULT_TRANSMISSION, nuisance=False)
    config = next(c for c in fsm.build_configs(False) if c.key == "nfix_cr1")
    residual = fsm.make_profiled_residual(dep, config, data)
    theta_vec = np.array([12.0, 0.8])  # arbitrary interior point
    theta_full = residual.recover_full(theta_vec)
    chi2_profiled = float(np.sum(residual(theta_vec) ** 2))
    # Perturbing A in either direction must increase chi2.
    for factor in (0.999, 1.001):
        perturbed = dict(theta_full)
        perturbed["A"] = theta_full["A"] * factor
        chi2 = float(
            np.sum(fsm.residuals_full(dep, config, data, perturbed) ** 2)
        )
        assert chi2 >= chi2_profiled - 1e-12, (
            f"profiled A not optimal: {chi2:.9f} < {chi2_profiled:.9f}"
        )
    # And for the two-amplitude (C_r-free) variant.
    config_free = next(
        c for c in fsm.build_configs(False) if c.key == "nfree"
    )
    residual_free = fsm.make_profiled_residual(dep, config_free, data)
    theta_free = residual_free.recover_full(theta_vec)
    chi2_free = float(np.sum(residual_free(theta_vec) ** 2))
    for name in ("A", "cr"):
        for factor in (0.999, 1.001):
            perturbed = dict(theta_free)
            perturbed[name] = theta_free[name] * factor
            chi2 = float(
                np.sum(
                    fsm.residuals_full(dep, config_free, data, perturbed) ** 2
                )
            )
            assert chi2 >= chi2_free - 1e-12, f"nfree {name} not optimal"


TESTS = [
    test_parser_round_trip,
    test_energy_conservation,
    test_full_model_limits,
    test_m2_matches_full_model_to_first_order,
    test_back_escape_integral,
    test_continuity_in_u,
    test_variable_projection,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        name = test.__name__
        try:
            test()
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"pass  {name}")
    print(
        f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed"
        + ("" if failures == 0 else f", {failures} FAILED")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
