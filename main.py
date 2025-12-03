#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce the paper's humid-air absorption + dispersion around 1–2 µm
using HITRAN line data (H2O) with the same assumptions:

- Temperature: 22 °C (295 K)
- Pressure: 1 atm
- Lineshape: Voigt
- Spectral grid resolution: 5e-3 cm^-1  (paper)
- Humidity cases (absolute humidity at 22 °C): 3.1, 6.2, 12.4 g/m^3
  which correspond to ~15%, 30%, 60% RH in their chamber measurement
- Example propagation: L = 12 m

Outputs:
- CSV with wavenumber [cm^-1], wavelength [nm], alpha(cm^-1) and Δn for each humidity
- PNG plots for quick inspection
- Demo linear propagation of a 330 fs-class spectrum near 1910 nm (replace with your own)

Notes:
- Based on the paper’s method: scale atmospheric transmission with HITRAN,
  then KK to get Δn, then linear propagation E_out(ω) = E_in(ω) * exp[i ω Δn z / c - α z / 2].
  See their Eq. (1), (2) and parameter list.  [oai_citation:2‡oe-23-11-13776.pdf](sediment://file_0000000044f47246bf438eeb70048fd6)
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy only for Voigt and Hilbert machinery in KK
from scipy.special import wofz
from scipy.signal import hilbert

# ---- HITRAN (HAPI) ----------------------------------------------------------
# pip install hitran-api
try:
    from hapi import db_begin, fetch, absorptionCoefficient_Voigt, partitionSum
except Exception as e:
    raise SystemExit("Please install HAPI (HITRAN API):  pip install hitran-api\n" + str(e))

# ----------------------------- constants -------------------------------------
c_cm_s = 2.99792458e10      # speed of light [cm/s]
k_B    = 1.380649e-23       # J/K
atm_Pa = 101325.0           # Pa
R_spec_H2O = 461.5          # J/(kg·K)
pi = math.pi

# ----------------------------- helpers ---------------------------------------
def ah_gm3_to_p_atm(ah_gm3, T_K):
    """
    Convert absolute humidity ρ [g/m^3] at temperature T to partial pressure p_H2O [atm]
    via ideal gas: p = ρ * R_specific * T   (R_specific water vapor = 461.5 J/kg/K).
    """
    rho = ah_gm3 * 1e-3  # kg/m^3
    p_Pa = rho * R_spec_H2O * T_K   # Pa
    return p_Pa / atm_Pa

def number_density_cm3(p_atm, T_K):
    """Molecules per cm^3 from partial pressure p_atm and T using ideal gas."""
    n_m3 = (p_atm * atm_Pa) / (k_B * T_K)     # [1/m^3]
    return n_m3 / 1e6                          # [1/cm^3]

def voigt_profile(nu, nu0, sigmaD, gammaL):
    """
    Normalized Voigt in wavenumber domain [cm^-1].
    sigmaD: Doppler std dev in cm^-1, gammaL: Lorentz HWHM in cm^-1.
    """
    # Convert to dimensionless variables for Faddeeva
    x = (nu - nu0) / (sigmaD * np.sqrt(2))
    y = gammaL / (sigmaD * np.sqrt(2))
    z = x + 1j*y
    V = np.real(wofz(z)) / (sigmaD * np.sqrt(2*np.pi))
    return V  # area-normalized to 1

def doppler_sigma_cm1(nu0_cm1, T_K, molar_mass_gmol=18.01528):
    """
    Doppler sigma [cm^-1]; nu0 in cm^-1. Uses sqrt(kT/m) with m from molar mass.
    """
    # Doppler FWHM in frequency is: Δν_D = ν0 * sqrt(8 kT ln2 / (mc^2))
    # Convert to wavenumber sigma:
    m_kg = molar_mass_gmol / 1000.0 / 6.02214076e23  # kg per molecule
    sigma = nu0_cm1 * np.sqrt(k_B*T_K/(m_kg * (c_cm_s*1e-2)**2))  # rough, but fine
    # sigma here is not exactly std dev; for Voigt we need std dev in wavenumber units.
    # The above gives σ ≈ ν0 * sqrt(kT/(mc^2)). Good enough for Voigt here.
    return sigma

def kk_delta_n_from_alpha(nu_cm1, alpha_cm1, n_infty=1.0):
    """
    KK to get Δn(ν) from α(ν), using κ(ν) = α / (4πν) and the dispersion relation
    in wavenumber form:

      n(ν) - n(∞) = (2/π) P ∫_0^∞ [ ν' κ(ν') / (ν'^2 - ν^2) ] dν'

    Numerical treatment:
    - Implemented via the identity ν'/(ν'^2 - ν^2) = 0.5 * [ 1/(ν' - ν) - 1/(ν' + ν) ]
    - First term is a Hilbert transform of κ(ν); we approximate with FFT Hilbert on a
      zero-padded window. Second term is a one-sided convolution we compute directly.
    - Pad and window edges to reduce end effects (still, make your window comfortably wide).

    This mirrors the paper’s McLaurin-style practical approach for KK integration.  [oai_citation:3‡oe-23-11-13776.pdf](sediment://file_0000000044f47246bf438eeb70048fd6)
    """
    nu = nu_cm1.astype(float)
    kappa = alpha_cm1 / (4.0*pi*np.maximum(nu, 1e-12))
    dnu = nu[1] - nu[0]

    # zero-pad for Hilbert stability
    pad = 4 * len(nu)
    k_pad = np.pad(kappa, (pad, pad), mode='edge')
    # Hilbert transform on padded array
    H_full = np.imag(hilbert(k_pad))
    H = H_full[pad:-pad]  # de-pad

    # First piece: (1/π) * Hilbert[kappa]
    term1 = (1.0/pi) * H

    # Second piece: -(1/π) * ∫ kappa(ν')/(ν'+ν) dν'  (simple trapezoid)
    # Vectorized eval: for each ν_j, denominator is (ν + ν'), integrate over ν'
    # This is O(N^2); to keep runtime sane, downsample for the integral.
    # For exactness, set down=1. For speed, use down=2,4…
    down = 2
    nu_s = nu[::down]
    k_s  = kappa[::down]
    dnu_s = dnu * down

    # Build matrix 1/(ν'+ν) and integrate
    denom = (nu_s[None, :] + nu[:, None])  # shape (N, Ns)
    term2 = (1.0/pi) * np.sum(k_s[None, :] / denom, axis=1) * dnu_s

    delta_n = term1 - term2   # n(ν) - n(∞)
    return delta_n

def ensure_hapi(api_key_env='HITRAN_API_KEY', db_dir='hitran_db'):
    """Initialize local HAPI DB and authenticate if the installed HAPI supports it."""
    os.makedirs(db_dir, exist_ok=True)
    db_begin(db_dir)
    key = os.environ.get(api_key_env, '')
    # Newer HAPI auto-uses HITRAN API key from env; older prompts on first fetch.
    if not key:
        print("Note: environment variable HITRAN_API_KEY is not set. "
              "If HAPI prompts for credentials, provide your HITRANonline details.")

def fetch_h2o_lines(table_name, nu_min, nu_max):
    """
    Fetch H2O lines over [nu_min, nu_max] cm^-1. Try all isotopologues when possible.
    """
    try:
        # iso=0 means "all isotopologues" in newer HAPI; fallback to main iso (1)
        fetch(table_name, 1, 0, nu_min, nu_max)  # H2O molec_id=1
    except Exception:
        fetch(table_name, 1, 1, nu_min, nu_max)
    return table_name

def absorption_from_hitran(nu_grid, T_K, p_atm_total, x_h2o, table_name='H2O'):
    """
    Use HAPI to build α(ν) [cm^-1] on nu_grid with Voigt at (T, p).
    We compute cross sections in HITRAN units, then multiply by number density.
    This honors pressure broadening via Environment={'T','p','VMR'}.
    """
    # Cross section [cm^2/molecule], pressure-broadened with total p and self VMR
    # HAPI will use gamma_air, gamma_self etc. using (T,p,VMR).
    nu, sigma = absorptionCoefficient_Voigt(
        SourceTables=table_name,
        WavenumberGrid=nu_grid,
        Environment={'T': T_K, 'p': p_atm_total, 'VMR': {'H2O': x_h2o}},
        HITRAN_units=True
    )
    # number density of water
    n_cm3 = number_density_cm3(p_atm_total * x_h2o, T_K)
    alpha_cm1 = sigma * n_cm3  # [cm^-1]
    return alpha_cm1

# --------------------------- main config -------------------------------------
if __name__ == '__main__':
    # Spectral window: full 1–2 µm would be 5000 cm^-1 wide. That’s huge for KK.
    # For the paper’s 1.9 µm case, focus on ~5100–5400 cm^-1 (~1889–1961 nm),
    # still at the paper’s 5e-3 cm^-1 resolution.
    # You can widen to (5000, 10000) if you only need alpha(ν) and skip KK.
    NU_MIN = 5100.0
    NU_MAX = 5400.0
    DNU    = 5e-3  # paper’s resolution 5×10^-3 cm^-1   [oai_citation:4‡oe-23-11-13776.pdf](sediment://file_0000000044f47246bf438eeb70048fd6)

    T_K = 295.0        # 22 °C   [oai_citation:5‡oe-23-11-13776.pdf](sediment://file_0000000044f47246bf438eeb70048fd6)
    P_ATM = 1.0        # 1 atm   [oai_citation:6‡oe-23-11-13776.pdf](sediment://file_0000000044f47246bf438eeb70048fd6)
    L_m = 12.0         # path length 12 m   [oai_citation:7‡oe-23-11-13776.pdf](sediment://file_0000000044f47246bf438eeb70048fd6)

    # Absolute humidities from the paper: 3.1, 6.2, 12.4 g/m^3 (≈ 15/30/60 %RH at 22 °C).
    # They used these inside the climate chamber.  [oai_citation:8‡oe-23-11-13776.pdf](sediment://file_0000000044f47246bf438eeb70048fd6)
    AH_LIST = [3.1, 6.2, 12.4]  # g/m^3

    OUTDIR = 'air_1to2um_outputs'
    os.makedirs(OUTDIR, exist_ok=True)

    # Grid
    nu_grid = np.arange(NU_MIN, NU_MAX + DNU, DNU)  # [cm^-1]
    wl_nm   = 1e7 / nu_grid  # nm

    # Init HAPI + fetch lines
    ensure_hapi()
    table = fetch_h2o_lines('H2O_1910nm', NU_MIN, NU_MAX)

    # Compute α and Δn for each humidity
    results = []
    for ah in AH_LIST:
        p_h2o = ah_gm3_to_p_atm(ah, T_K)   # atm
        x_h2o = p_h2o / P_ATM
        alpha = absorption_from_hitran(nu_grid, T_K, P_ATM, x_h2o, table_name=table)

        # KK for Δn (relative ripple; absolute offset n(∞) set to 1)
        delta_n = kk_delta_n_from_alpha(nu_grid, alpha, n_infty=1.0)

        # Save CSV
        df = pd.DataFrame({
            'nu_cm^-1': nu_grid,
            'wavelength_nm': wl_nm,
            f'alpha_cm^-1_AH_{ah:g}': alpha,
            f'delta_n_AH_{ah:g}': delta_n
        })
        csv_path = os.path.join(OUTDIR, f'abs_disp_AH_{ah:g}_g_per_m3.csv')
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

        results.append((ah, p_h2o, x_h2o, alpha, delta_n))

    # Quick plots
    for ah, p_h2o, x_h2o, alpha, delta_n in results:
        plt.figure()
        plt.plot(wl_nm, alpha)
        plt.gca().invert_xaxis()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Absorption coefficient α [cm$^{-1}$]')
        plt.title(f'H₂O absorption (AH {ah} g m$^{{-3}}$, T=22°C, p=1 atm)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'alpha_AH_{ah:g}.png'), dpi=180)
        plt.close()

        plt.figure()
        plt.plot(wl_nm, delta_n)
        plt.gca().invert_xaxis()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Δn (KK from α)')
        plt.title(f'H₂O-induced refractive index ripple (AH {ah} g m$^{{-3}}$)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'delta_n_AH_{ah:g}.png'), dpi=180)
        plt.close()

    # ------------------------- demo: linear propagation ------------------------
    # Paper’s oscillator: centered ~1910 nm, ~330 fs after compression.  [oai_citation:9‡oe-23-11-13776.pdf](sediment://file_0000000044f47246bf438eeb70048fd6)
    # We build a Gaussian E(ν) just to show the effect; replace with your real spectrum.
    center_nm = 1910.0
    # Choose a spectral sigma to land near a few-hundred-fs in time
    sigma_nm = 12.0
    Ein = np.exp(-0.5*((wl_nm - center_nm)/sigma_nm)**2).astype(np.complex128)

    # Choose the wettest case (worst distortions)
    ah, p_h2o, x_h2o, alpha, delta_n = results[-1]

    # Transfer function in wavenumber ν:
    # k = 2π ν; phase advance from Δn is 2π ν Δn L; amplitude loss exp(-α L/2)
    L_cm = L_m * 100.0
    H = np.exp(1j * 2.0*pi*nu_grid*delta_n*L_cm) * np.exp(-alpha * L_cm / 2.0)

    Eout = Ein * H

    # Time-domain view via FFT (uniform grid in ν -> non-uniform in ω; this is heuristic)
    # Use wavenumber-domain FFT for a qualitative post-pulse picture.
    # If you want quantitatively correct timing, do it in angular frequency.
    Etd_in  = np.fft.ifft(np.fft.ifftshift(Ein))
    Etd_out = np.fft.ifft(np.fft.ifftshift(Eout))
    t = np.arange(len(nu_grid))  # arbitrary samples

    plt.figure()
    plt.plot(t, np.abs(Etd_in)/np.max(np.abs(Etd_in)), label='in')
    plt.plot(t, np.abs(Etd_out)/np.max(np.abs(Etd_out)), label='out (60% RH case)')
    plt.xlabel('Arbitrary time samples')
    plt.ylabel('Normalized |E(t)|')
    plt.title('Demo: post-pulses from humid-air lines (qualitative)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'demo_time_domain.png'), dpi=180)
    plt.close()

    print("Done. CSVs and PNGs in:", OUTDIR)