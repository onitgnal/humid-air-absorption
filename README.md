# Humid Air Absorption and Refractive Index Ripple Simulation

This repository contains a script (`plot_transmission.py`) designed to simulate and plot the atmospheric absorption (or transmission) and the corresponding refractive-index ripple (Δn) caused by water vapor across a specified infrared wavelength window.

## Code Description

The script performs the following key steps:
1. **Atmospheric Modeling:** It calculates the water vapor number density based on the specified temperature (22°C default), relative humidity, and pressure (1 atm default) using the Tetens formula for saturation vapor pressure.
2. **HITRAN Data Fetching:** It uses the HAPI (HITRAN Application Programming Interface) library to download and cache water-vapor absorption line data for the requested wavenumber range.
3. **Absorption Profile (α):** It computes the highly resolved absorption coefficient α(ν) across a wavenumber grid using a Voigt line profile, which accounts for both Doppler (thermal) and pressure broadening.
4. **Refractive Index Ripple (Δn):** It calculates the change in the real part of the refractive index (Δn) induced by these absorption resonances using Kramers-Kronig (KK) relations.
5. **Output:** It exports the calculated wavelength, transmission, Δn, and optical phase shift to a high-resolution text file (`trans_delta_n_fullres.txt`) and generates a dual-axis plot (`trans_and_delta_n.png`) visualizing either transmission or absorption alongside Δn.

## Source from the Literature

The physical and numerical methodology—specifically how the refractive index changes (Δn) are derived from the absorption lines via Kramers-Kronig relations—is based on the following paper:

*   **Gebhardt et al., "Impact of atmospheric molecular absorption on the temporal and spatial evolution of ultra-short optical pulses," *Optics Express* 23(11):13776-13787 (2015). DOI: [10.1364/OE.23.013776](https://doi.org/10.1364/OE.23.013776)**

## Usage and Arguments

The script is executed via the command line. You can customize the simulation using the following arguments:

```bash
python plot_transmission.py [arguments]
```

**Physics & Output Arguments:**
*   `--lambda-min-um` (float): Start wavelength of the window in µm. **Default:** `1.0`
*   `--lambda-max-um` (float): End wavelength of the window in µm. **Default:** `2.0`
*   `--dnu` (float): Spectral resolution/step size in wavenumbers (cm⁻¹). **Default:** `0.05`
*   `--path-cm` (float): Optical path length in centimeters. Used to calculate the displayed transmission or total absorption. **Default:** `1.0`
*   `--rh` (float): Relative humidity as a fraction (0.0 to 1.0). **Default:** `0.50` (50%)
*   `--y-mode` (string): What to plot on the primary Y-axis. Choices: `transmission` or `alpha` (absorption coefficient in m⁻¹). **Default:** `transmission`
*   `--delta-n-unit` (string): Unit for the Δn axis. Choices: `dimensionless` or `ppm` (parts per million). **Default:** `dimensionless`
*   `--delta-n-sign` (string): Forces the sign convention for Δn. `paper` matches the Gebhardt reference; `previous` flips the sign. **Default:** `paper`

**Numerical Method Arguments:**
*   `--kk-method` (string): The integration method for the Kramers-Kronig relation. Choices: `maclaurin` or `hilbert`. **Default:** `maclaurin`
*   `--kk-downsample` (int): *Only applies to `hilbert`*. Downsamples the slow integral correction term to speed up computation at the cost of accuracy. **Default:** `1`
*   `--kk-pad-factor` (int): *Only applies to `hilbert`*. Zero-padding multiplier for the FFT to reduce edge artifacts. **Default:** `4`

## Evaluation of Kramers-Kronig (KK) Methods

The Kramers-Kronig relation is an integral equation that relates the real part of the refractive index to the imaginary part (absorption). Because it involves a Cauchy principal value (an integral containing a singularity where the integration variable $\omega_j$ equals the frequency of interest $\omega_i$), calculating it numerically on finite discrete data is challenging.

The script offers two approaches:

### The `maclaurin` Method (Recommended & Consistent with Literature)
This method (`kk_delta_n_maclaurin`) uses the **Maclaurin quadrature**, specifically the Ohta–Ishida formula, to directly perform a numerical integration of the principal-value integral:

$$ \Delta n(\omega_i) \approx \frac{c}{\pi} 2h \sum' \frac{\alpha_j}{\omega_j^2 - \omega_i^2} $$

*   **Why it is accurate and consistent:** To avoid the mathematical singularity (division by zero) when $\omega_j = \omega_i$, this algorithm cleverly sums only over alternating grid points ($\Sigma'$ signifies summing over points with the opposite parity to $i$). This naturally interpolates exactly over the pole. 
*   **Consistency with the Paper:** The Gebhardt et al. paper explicitly calculates the refractive index by directly evaluating the analytical KK sums/integrals over the absorption resonances. The `maclaurin` quadrature is the mathematically rigorous way to evaluate these specific Cauchy principal-value integrals on a uniform discrete grid. It avoids the FFT windowing artifacts entirely, ensuring that the local phase shifts match the analytical theory described in the literature.

### The `hilbert` Method
This method (`kk_delta_n_from_alpha`) uses `scipy.signal.hilbert` (an FFT-based discrete Hilbert transform) combined with a slow secondary summation term to approximate the KK integral. 
*   **Drawbacks:** FFT-based Hilbert transforms assume the signal is periodic and infinite. When applied to a finite, non-periodic spectral window of absorption lines, it suffers heavily from **edge effects** (spectral leakage at the boundaries). While padding (`--kk-pad-factor`) mitigates this, the discrete nature of the FFT often struggles to accurately capture the sharp, narrow local phase ripples caused by individual molecular resonances without introducing global baseline offsets.
