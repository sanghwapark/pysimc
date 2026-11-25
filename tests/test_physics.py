import numpy as np
import pytest

from pysimc.config import BeamConfig, TargetConfig
from pysimc.physics import bosted_form_factors, mott_xsec
from pysimc.target import luminosity_per_ub


def test_mott_known_value():
    # alpha^2 (hbar c)^2 cos^2(30) / (4 E^2 sin^4(30)) at E=1 GeV, 60 deg
    # = 6.2e-6 fm^2/sr = 0.062 ub/sr
    assert mott_xsec(1000.0, np.deg2rad(60.0)) == pytest.approx(0.0622, rel=5e-3)


def test_bosted_matches_fortran():
    # fofa_best_fit(-Q² / hbarc², GE, GM), compiled from SIMC revision 60c2047.
    q2_mev2, ge, gm = np.array([
        (0.0,    1.0,                  2.793),
        (0.25e6, 0.5313849219528396,   1.4523236919077023),
        (1e6,    0.1686340640809444,   0.4925925925925926),
        (8e6,    0.008021150656990272, 0.01735534685650736),
    ]).T
    np.testing.assert_allclose(bosted_form_factors(q2_mev2), (ge, gm), rtol=1e-12, atol=0)


def test_bosted_rejects_timelike_q2():
    with pytest.raises(ValueError, match="spacelike"):
        bosted_form_factors(-1.0)


def test_luminosity_matches_fortran_constant():
    beam = BeamConfig(charge_mC=1.0)
    tgt = TargetConfig(mass_amu=1.007276, thickness_mg_cm2=295.172, abundancy_pct=100.0)
    fortran = 1.0 / (tgt.mass_amu / 3.75914e6 / 1.0 / (tgt.thickness_mg_cm2))
    assert luminosity_per_ub(beam, tgt) == pytest.approx(fortran, rel=2e-3)


def test_luminosity_scales_with_tilt():
    beam = BeamConfig(charge_mC=2.0)
    flat = luminosity_per_ub(beam, TargetConfig(angle_deg=0.0))
    tilt = luminosity_per_ub(beam, TargetConfig(angle_deg=60.0))
    assert tilt == pytest.approx(2 * flat)
