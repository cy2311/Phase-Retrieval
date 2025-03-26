import numpy as np
from scipy.special import factorial
from typing import Tuple, Dict, Optional, Union

ZernikeCoeffs = Dict[str, Union[int, float, list, np.ndarray]]
ZernikeBasis = Dict[str, Union[np.ndarray, Dict[int, list]]]

def create_zernike_state(ordering: str = 'Wyant') -> ZernikeCoeffs:
    # Create zernike polynomials calculation status dictionary
    return {
        'ordering': ordering,
        'w_abs_err': 1.0e-01,
        'w_rel_err': 1.0e-01,
        'basis_matrix': None,  # ZM
        'max_order': -1,       # Wnmax
        'max_index': -1,       # Wlmax
        'coefficients': [],    # WZc
        'normalization_radius': 1.0
    }

# Coefficient calculate core function
def compute_wyant_coefficients(n: int, m: int) -> np.ndarray:
    # Calculate Wyant ordering zernike polynomials coefficients
    num_terms = n - m + 1
    c = np.zeros(num_terms)
    for k in range(num_terms):
        start = n - m - k + 1
        end = 2 * n - m - k
        numerator = (-1)**k * np.prod(np.arange(start, end + 1)) if start <= end else 1.0
        denominator = factorial(k) * factorial(n - k)
        c[k] = numerator / denominator
    return c

def initialize_wyant_coefficients(state: ZernikeCoeffs) -> ZernikeCoeffs:
    # Initialize Wyant ordering zernike polynomial coefficients without status changing
    state = state.copy()  
    
    if state['max_index'] == -1:
        if state['max_order'] == -1:
            raise ValueError('Both max_index and max_order undefined!')
        state['max_index'] = (state['max_order'] + 1)**2 - 1
    else:
        if state['max_order'] == -1:
            state['max_order'] = int(np.ceil(np.sqrt(state['max_index'] + 1)) - 1)
        else:
            max_possible = (state['max_order'] + 1)**2 - 1
            if state['max_index'] > max_possible:
                raise ValueError(f'Index {state["max_index"]} too large for order {state["max_order"]}')

    state['coefficients'] = []
    for n in range(state['max_order'] + 1):
        n_coeffs = []
        for m in range(n + 1):
            coeff = compute_wyant_coefficients(n, m)
            n_coeffs.append(coeff)
        state['coefficients'].append(n_coeffs)
    
    return state


# Polynomial calculation function
def evaluate_wyant_polynomial(
    state: ZernikeCoeffs,
    n: int,
    m: int,
    rho: np.ndarray,
    theta: np.ndarray
) -> np.ndarray:
    # Calculate specific (n, m) zernike polynomial values
    if n >= len(state['coefficients']):
        raise ValueError(f'Insufficient coefficients for n={n}')
    
    mm = abs(m)
    c = state['coefficients'][n][mm]
    
    # Using Horner method calculate poynomials
    p = np.zeros_like(rho)
    rho_power = rho**mm
    rho_sq = rho**2
    
    for k in reversed(range(len(c))):
        p = p * rho_sq + c[k]
    
    p *= rho_power
    
    if m > 0:
        return p * np.cos(mm * theta)
    elif m < 0:
        return p * np.sin(mm * theta)
    return p

# Basic matrix generation

def generate_basis_matrix(
    state: ZernikeCoeffs,
    rho: np.ndarray,
    theta: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[ZernikeCoeffs, np.ndarray]:
    state = initialize_wyant_coefficients(state)
    state = state.copy()
    
    if np.any((rho < 0) | (rho > 1)):
        raise ValueError("rho must be in [0,1]")
    
    n_max = state['max_order']
    R, C = rho.shape
    n_basis = (n_max + 1)**2
    
    # pre-calculate rho 
    max_power = 2 * n_max
    rho_powers = np.zeros((max_power + 1, R, C))
    rho_powers[0] = 1.0
    for i in range(1, max_power + 1):
        rho_powers[i] = rho_powers[i-1] * rho
    
    # initialize basic matrix 
    Z = np.zeros((R, C, n_basis))
    if mask is None:
        mask = np.ones((R, C), dtype=bool)
    
    Z[..., 0] = mask.astype(float)
    
    basis_index = 1
    for n in range(1, n_max + 1):
        for m in range(n, -1, -1):
            if m >= len(state['coefficients'][n]):
                continue
            
            c = state['coefficients'][n][m]
            exponents = np.arange(2*n - m, m - 1, -2)
            
            # rho polynomial function calculation
            radial = np.zeros_like(rho)
            for k, exp in enumerate(exponents):
                radial += c[k] * rho_powers[exp]
            
            # add angle items
            if m == 0:
                Z[..., basis_index] = radial * mask
                basis_index += 1
            else:
                Z[..., basis_index] = radial * np.cos(m * theta) * mask
                Z[..., basis_index+1] = radial * np.sin(m * theta) * mask
                basis_index += 2
            
            if basis_index >= n_basis:
                break
    
    state['basis_matrix'] = Z
    return state, Z

# fitness function

def fit_zernike(
    state: ZernikeCoeffs,
    pupil: np.ndarray,
    max_order: int,
    norm_radius: float = 1.0,
    dtype: type = np.float64
) -> Tuple[np.ndarray, np.ndarray]:
    if state['basis_matrix'] is None:
        raise ValueError("Basis matrix not computed")
    
    n_terms = (max_order + 1)**2
    Z = state['basis_matrix'][..., :n_terms].astype(dtype)
    
    coeffs = np.zeros(n_terms, dtype=dtype)
    for k in range(n_terms):
        Zk = Z[..., k]
        numer = np.sum(pupil * Zk)
        denom = np.sum(Zk * Zk)
        if denom > 1e-12:
            coeffs[k] = numer / denom
    
    fitted = np.zeros_like(pupil, dtype=dtype)
    for k in range(n_terms):
        fitted += Z[..., k] * coeffs[k]
    
    if np.iscomplexobj(pupil):
        norm = np.sqrt(np.sum(np.abs(fitted)**2) + 1e-12)
        coeffs /= norm
        fitted /= norm
    
    return coeffs, fitted

def wyant_index_to_nm(index: int) -> Tuple[int, int]:
    """Wyant索引转换为(n,m)"""
    n = int(np.floor(np.sqrt(index)))
    mm = int(np.ceil((2*n - (index - n**2)) / 2))
    return (n, mm) if (index - n**2) % 2 == 0 else (n, -mm)

def nm_to_wyant_index(n: int, m: int) -> int:
    """(n,m)转换为Wyant索引"""
    return n**2 + 2*(n - abs(m)) + (0 if m >=0 else 1)


def calculate_polar_coordinates(
    image_size: int,
    pixel_size: float,
    magnification: float,
    na: float,
    wavelength: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(-image_size/2, image_size/2-1, image_size)
    x = np.linspace(-image_size/2, image_size/2-1, image_size)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    Z = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    
    scale = image_size * pixel_size / magnification
    max_freq = na / wavelength
    k_r = Z / scale
    
    na_mask = k_r < max_freq
    rho = np.where(na_mask, k_r / max_freq, 0)
    theta = np.where(na_mask, PHI, 0)
    
    return rho, theta, na_mask, k_r
