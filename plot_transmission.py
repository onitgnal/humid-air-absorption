#!/usr/bin/env python3
"""
Plot atmospheric transmission and refractive-index ripple (Δn from KK of α)
over a configurable wavelength window using HITRAN (HAPI) water-vapor data.
Default: 1–2 µm, sea level, 22 °C, 50% RH, 1 cm path.
"""

import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.signal import hilbert
from hapi import db_begin, fetch, absorptionCoefficient_Voigt

# Physical constants
ATM_PA = 101325.0
K_B = 1.380649e-23


def ensure_hapi(db_dir="hitran_db"):
    """Initialize local HAPI database directory."""
    os.makedirs(db_dir, exist_ok=True)
    db_begin(db_dir)


def saturation_vapor_pressure_atm(T_C):
    """
    Saturation vapor pressure [atm] at temperature T_C (Celsius).
    Tetens formula (reasonable around room temperature).
    """
    Pws_kPa = 0.61094 * np.exp(17.625 * T_C / (T_C + 243.04))
    return (Pws_kPa * 1000.0) / ATM_PA


def number_density_cm3(p_atm, T_K):
    """Molecules per cm^3 from partial pressure p_atm and temperature."""
    n_m3 = (p_atm * ATM_PA) / (K_B * T_K)
    return n_m3 / 1e6


def fetch_h2o_lines(table, nu_min, nu_max):
    """Fetch H2O lines over the given wavenumber window."""
    try:
        fetch(table, 1, 0, nu_min, nu_max)  # iso=0 for all isotopologues if supported
    except Exception:
        fetch(table, 1, 1, nu_min, nu_max)  # fallback to main isotopologue
    return table


def absorption_alpha(nu_grid, T_K, p_atm_total, x_h2o, table):
    """
    Absorption coefficient α(ν) [cm^-1] for water vapor on nu_grid using HAPI Voigt.
    """
    nu, sigma = absorptionCoefficient_Voigt(
        SourceTables=table,
        WavenumberGrid=nu_grid,
        Environment={"T": T_K, "p": p_atm_total, "VMR": {"H2O": x_h2o}},
        HITRAN_units=True,
    )
    n_cm3 = number_density_cm3(p_atm_total * x_h2o, T_K)
    return sigma * n_cm3


def kk_delta_n_from_alpha(nu_cm1, alpha_cm1, downsample=1, pad_factor=4):
    """
    Kramers-Kronig to get refractive index ripple Δn(ν) from absorption α(ν).
    Uses a Hilbert transform plus a secondary integral term. Downsample the slow
    integral for speed (set downsample=1 for maximum accuracy).
    """
    pi = math.pi
    nu = nu_cm1.astype(float)
    dnu = nu[1] - nu[0]
    kappa = alpha_cm1 / (4.0 * pi * np.maximum(nu, 1e-12))

    pad = pad_factor * len(nu)
    k_pad = np.pad(kappa, (pad, pad), mode="edge")
    H_full = np.imag(hilbert(k_pad))
    H = H_full[pad:-pad]
    term1 = (1.0 / pi) * H

    nu_s = nu[::downsample]
    k_s = kappa[::downsample]
    dnu_s = dnu * downsample
    denom = nu_s[None, :] + nu[:, None]
    term2 = (1.0 / pi) * np.sum(k_s[None, :] / denom, axis=1) * dnu_s

    delta_n = term1 - term2
    return delta_n


def kk_delta_n_maclaurin(nu_cm1: np.ndarray, alpha_cm1: np.ndarray) -> np.ndarray:
    """
    Maclaurin principal-value KK on a UNIFORM ν-grid using the Ohta–Ishida /
    Gebhardt form in angular frequency:

      Δn(ω_i) ≈ (c/π) * 2h * Σ' [ α_j / (ω_j^2 - ω_i^2) ]

    where Σ' sums every other point (opposite parity to i) to skip the pole.
    alpha_cm1 is converted to m^-1 internally; ω = 2π c ν.
    """
    nu = np.asarray(nu_cm1, dtype=float)
    alpha_cm = np.asarray(alpha_cm1, dtype=float)
    if nu.ndim != 1 or alpha_cm.ndim != 1 or nu.size != alpha_cm.size:
        raise ValueError("nu_cm1 and alpha_cm1 must be 1D arrays of equal length")
    if nu.size < 3:
        raise ValueError("Need at least 3 points for Maclaurin quadrature")

    # Ensure uniform spacing
    h_nu = nu[1] - nu[0]
    if not np.allclose(np.diff(nu), h_nu, rtol=0, atol=1e-10 * abs(h_nu)):
        raise ValueError("Maclaurin requires a UNIFORM wavenumber grid")

    # Convert to angular frequency ω (rad/s); h in ω-domain
    c_m_s = 299792458.0
    omega = 2.0 * np.pi * c_m_s * (nu * 100.0)  # nu in cm^-1 -> m^-1
    h_omega = omega[1] - omega[0]
    omega_sq = omega**2

    # α to m^-1
    alpha_m = alpha_cm * 100.0

    m = nu.size
    dn = np.zeros_like(nu)
    idx_all = np.arange(m)

    factor = (c_m_s / np.pi) * 2.0 * h_omega

    for i in range(m):
        if (i % 2) == 0:
            jj = idx_all[1::2]
        else:
            jj = idx_all[0::2]
        jj = jj[jj != i]

        denom = omega_sq[jj] - omega_sq[i]
        dn[i] = factor * np.sum(alpha_m[jj] / denom)

    if m >= 3:
        dn[0] = dn[1]
        dn[-1] = dn[-2]
    return dn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot atmospheric transmission for water vapor. "
            'Reference: Gebhardt et al., "Impact of atmospheric molecular absorption on the temporal '
            'and spatial evolution of ultra-short optical pulses," Opt. Express 23(11):13776-13787 (2015), '
            "doi:10.1364/OE.23.013776."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lambda-min-um", type=float, default=1.0, help="Start wavelength in µm.")
    parser.add_argument("--lambda-max-um", type=float, default=2.0, help="End wavelength in µm.")
    parser.add_argument("--dnu", type=float, default=0.05, help="Wavenumber step in cm^-1.")
    parser.add_argument("--path-cm", type=float, default=1.0, help="Path length in cm for transmission/absorption plots.")
    parser.add_argument(
        "--rh",
        type=float,
        default=0.50,
        help="Relative humidity (0.0 to 1.0). Default is 0.50.",
    )
    parser.add_argument(
        "--y-mode",
        choices=["transmission", "alpha"],
        default="transmission",
        help="Plot transmission for the given path (uses --path-cm) or absorption coefficient in m^-1.",
    )
    parser.add_argument(
        "--delta-n-unit",
        choices=["dimensionless", "ppm"],
        default="dimensionless",
        help="Display Δn as-is or scaled to ppm.",
    )
    parser.add_argument(
        "--delta-n-sign",
        choices=["paper", "previous"],
        default="paper",
        help="Use the sign convention matching the paper ('paper') or flip to the previous code ('previous').",
    )
    parser.add_argument(
        "--kk-method",
        choices=["maclaurin", "hilbert"],
        default="maclaurin",
        help="KK method: maclaurin (principal-value sum) or hilbert (FFT-based).",
    )
    parser.add_argument(
        "--kk-downsample",
        type=int,
        default=1,
        help="Downsample factor for the slow KK integral (1 = full fidelity; >1 speeds up but can under-estimate |Δn|).",
    )
    parser.add_argument(
        "--kk-pad-factor",
        type=int,
        default=4,
        help="Zero-padding factor for the Hilbert transform (reduce edge effects).",
    )
    args = parser.parse_args()

    # Conditions
    T_C = 22.0
    T_K = 273.15 + T_C
    RH = args.rh
    P_ATM = 1.0
    L_CM = args.path_cm

    # Spectral window
    NU_MAX = 1e4 / args.lambda_min_um  # cm^-1
    NU_MIN = 1e4 / args.lambda_max_um  # cm^-1
    DNU = args.dnu

    nu_grid = np.arange(NU_MIN, NU_MAX + DNU, DNU)
    wl_um = 1e4 / nu_grid

    # Humidity -> partial pressure -> mixing ratio
    p_sat_atm = saturation_vapor_pressure_atm(T_C)
    p_h2o_atm = RH * p_sat_atm
    x_h2o = p_h2o_atm / P_ATM

    progress = tqdm(total=5, desc="Processing", unit="step")

    progress.set_postfix_str("init HAPI")
    ensure_hapi()
    progress.update(1)

    progress.set_postfix_str("fetch HITRAN lines")
    table = fetch_h2o_lines("H2O_custom_window", NU_MIN, NU_MAX)
    progress.update(1)

    progress.set_postfix_str("absorption")
    alpha = absorption_alpha(nu_grid, T_K, P_ATM, x_h2o, table)
    progress.update(1)

    progress.set_postfix_str(f"KK Δn ({args.kk_method})")
    if args.kk_method == "maclaurin":
        delta_n = kk_delta_n_maclaurin(nu_grid, alpha)
    else:
        delta_n = kk_delta_n_from_alpha(
            nu_grid,
            alpha,
            downsample=max(1, args.kk_downsample),
            pad_factor=max(1, args.kk_pad_factor),
        )
    if args.delta_n_sign == "previous":
        delta_n = -delta_n
    progress.update(1)

    if args.delta_n_unit == "ppm":
        delta_n_plot = delta_n * 1e6
        delta_n_label = "Δn [ppm]"
    else:
        delta_n_plot = delta_n
        delta_n_label = "Δn"

    transmission = np.exp(-alpha * L_CM)
    transmission_1cm = np.exp(-alpha * 1.0)
    delta_n_ppm = delta_n * 1e6
    alpha_m = alpha * 100.0  # cm^-1 -> m^-1

    # Sort by wavelength for a left-to-right increasing axis
    order = np.argsort(wl_um)
    wl_sorted = wl_um[order]
    trans_sorted = transmission[order]
    trans_sorted_1cm = transmission_1cm[order]
    delta_n_sorted = delta_n[order]
    dn_sorted = delta_n_plot[order]
    dn_sorted_ppm = delta_n_ppm[order]
    alpha_m_sorted = alpha_m[order]

    L_M = 0.01  # 1 cm in meters
    wavelength_m = wl_sorted * 1e-6
    phase_sorted = (2.0 * math.pi / wavelength_m) * delta_n_sorted * L_M

    data_path = "trans_delta_n_fullres.txt"
    data_cols = np.column_stack((wl_sorted, trans_sorted_1cm, dn_sorted_ppm, phase_sorted))
    np.savetxt(
        data_path,
        data_cols,
        header="wavelength_um transmission_L1cm delta_n_ppm phase_rad_L1cm",
        fmt="%.10g",
    )

    fig, ax1 = plt.subplots(figsize=(8, 4))

    if args.y_mode == "transmission":
        y_values = trans_sorted
        ax1.plot(wl_sorted, y_values, color="navy", label=f"T (L={L_CM:.2f} cm)")
        ax1.set_ylabel(f"Transmission (L={L_CM:.2f} cm)")
    else:
        y_values = alpha_m_sorted
        ax1.plot(wl_sorted, y_values, color="navy", label="α [m^-1]")
        ax1.set_ylabel("Absorption coefficient α [m$^{-1}$]")

    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if math.isclose(y_min, y_max):
        pad = 0.05 * (abs(y_min) if y_min != 0 else 1.0)  # avoid zero-height axis when flat
        y_min -= pad
        y_max += pad
    ax1.set_ylim(y_min, y_max)

    ax1.set_xlabel("Wavelength [µm]")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(wl_sorted, dn_sorted, color="darkred", label=delta_n_label)
    ax2.set_ylabel(f"{delta_n_label} (from KK of α)")

    title = (
        f"Sea level, 50% RH, T=22°C | "
        f"{args.lambda_min_um:.2f}–{args.lambda_max_um:.2f} µm | Δn sign: {args.delta_n_sign}"
    )
    fig.suptitle(title)
    fig.tight_layout()

    plot_path = "trans_and_delta_n.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    progress.set_postfix_str("done")
    progress.update(1)
    progress.close()

    print(f"Saved combined plot to {plot_path}")
