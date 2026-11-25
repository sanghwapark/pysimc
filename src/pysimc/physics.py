"""Reusable physics formulae (vectorised).  Energies MeV, cross sections ub/sr.

Proton form-factor models take spacelike Q² in MeV² and return (GE, GM).
"""

from __future__ import annotations

import numpy as np

from .constants import ALPHA, FM2_TO_UB, HBARC, MP, MU_P


def mott_xsec(e_in, theta):
    """Point-like Mott cross section  alpha^2 cos^2(t/2) / (4 E^2 sin^4(t/2))  [ub/sr]."""
    s2 = np.sin(theta / 2.0) ** 2
    c2 = np.cos(theta / 2.0) ** 2
    return (ALPHA ** 2 * HBARC ** 2) * c2 / (4.0 * e_in ** 2 * s2 ** 2) * FM2_TO_UB


def dipole_form_factors(q2):
    """(GE, GM) of the proton, dipole model.  q2 in MeV^2."""
    gd = (1.0 + q2 / 0.71e6) ** -2
    return gd, MU_P * gd


def bosted_form_factors(q2):
    """Return the proton electric and magnetic form factors for spacelike Q² in MeV².

    P. E. Bosted, Phys. Rev. C 51, 409 (1995), https://doi.org/10.1103/PhysRevC.51.409.
    Eq. (4) fits GE up to Q² = 7 GeV². Eq. (5) fits GM / mu_p.
    Fortran source: ``fofa_best_fit`` in ``simc_gfortran/physics_proton.f``, line 137.
    """
    q2_gev2 = np.asarray(q2, dtype=float) * 1e-6  # Event Q² is in MeV², the fit uses GeV².
    if np.any(q2_gev2 < 0):
        raise ValueError("Bosted form factors require spacelike Q² >= 0")
    q = np.sqrt(q2_gev2)
    ge = 1.0 / (1.0 + 0.62*q + 0.68*q2_gev2 + 2.8*q**3 + 0.83*q**4)
    # Keep the magnetic moment used in the SIMC fit, including its rounding.
    gm = 2.793 / (1.0 + 0.35*q + 2.44*q2_gev2 + 0.5*q**3 + 1.04*q**4 + 0.34*q**5)
    return ge, gm


def rosenbluth_xsec(e_in, theta, q2, form_factors=dipole_form_factors, mass=MP):
    """Elastic e-p dsigma/dOmega_e [ub/sr], recoil included."""
    ge, gm = form_factors(q2)
    tau = q2 / (4.0 * mass ** 2)
    eprime = e_in / (1.0 + (e_in / mass) * (1.0 - np.cos(theta)))
    tan2 = np.tan(theta / 2.0) ** 2
    return mott_xsec(e_in, theta) * (eprime / e_in) * ((ge ** 2 + tau * gm ** 2) / (1.0 + tau) + 2.0 * tau * gm ** 2 * tan2)
