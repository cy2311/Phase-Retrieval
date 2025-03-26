# Phase-Retrieval
This Python code implements an iterative phase retrieval algorithm to reconstruct the pupil function and model astigmatic point spread functions (PSFs) from 3D microscopy data. It uses Zernike polynomials for aberration correction, Gaussian filtering for noise reduction, and parallel processing for efficiency

## Overview
This repository implements an Iterative Non-linear Spline Phase Retrieval (INSPR) algorithm for generating accurate point spread function (PSF) models from single-molecule localization microscopy data with astigmatism. The method combines phase retrieval techniques with Zernike polynomial aberration modeling to reconstruct the 3D PSF.

## Key Features
- Phase retrieval from astigmatic single-molecule images
- Zernike polynomial-based aberration modeling
- Iterative refinement of PSF models
- Parallel processing for efficient computation
- Drift correction and aberration compensation

## Dependencies
- Python 3.7+
- NumPy
- SciPy
- scikit-image
- joblib (for parallel processing)

## Installation
git clone https://github.com/yourusername/inspr-model-generator.git
cd inspr-model-generator
pip install -r requirements.txt

# Data Flow
## Input Data Preparation:
- Single-molecule subregions (64x64 pixels) stored in a 3D array (height × width × num_molecules)
- Microscope setup parameters (NA, wavelength, refractive index, etc.)
- Phase retrieval parameters (Z positions, iteration counts, etc.)
## Initialization Phase
graph TD
  A[Input Subregions] --> B[Initialize empupil parameters]
  B --> C[Generate Initial Pupil Function]
  C --> D[Create Reference PSFs]
## Iterative Refinement:
graph TD
  A[Current Pupil Estimate] --> B[Generate Aberrated PSFs]
  B --> C[Classify Single Molecules]
  C --> D[Average PSFs by Z-position]
  D --> E[Phase Retrieval Update]
  E --> F[Zernike Coefficient Estimation]
  F --> G[Drift Correction]
  G --> A
## Output:
- Final pupil function (magnitude and phase)
- Zernike polynomial coefficients
- 3D PSF model across specified Z-positions

# Key Components
1. Zernike Polynomial Handling
- create_zernike_state(): Initializes Zernike calculation state
- compute_wyant_coefficients(): Computes coefficients for Wyant ordering
- evaluate_wyant_polynomial(): Evaluates specific Zernike polynomials
- generate_basis_matrix(): Creates Zernike basis matrix

3. Phase Retrieval Core
- phaseretrieve(): Main phase retrieval algorithm
- PRPSF_aber_fromAveZ_ast(): Coordinates the iterative refinement process
- gen_aberPSF_fromPR_ast(): Generates aberrated PSFs from current pupil estimate

3. PSF Generation
- PSF_zernike class: Handles PSF generation with Zernike aberrations
- PSF_pupil class: Generates PSFs from pupil functions

4. Utility Functions
- FourierShift2D(): Performs subpixel shifts using Fourier transform
- classify_onePlane_par(): Parallel classification of single molecules to Z-positions
