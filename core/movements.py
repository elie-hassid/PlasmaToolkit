"""Definition of different motion laws."""

import numpy as np

def logarithmic(t: float, L0: float = 1e-3, a: float = 0.03, b: float = 2.06) -> float:
    """
    Logarithmic motion law.

    Parameters
    ----------
    t : float
        Time (s).
    L0 : float, optional
        Initial value (default 1e-3).
    a : float, optional
        Scaling coefficient (default 0.03).
    b : float, optional
        Growth factor for the logarithm (default 2.06).

    Returns
    -------
    float
        Position/length at time t.
    """
    return L0 + a * np.log(1 + b * t)

def linear(t: float, L0: float = 1e-3, v: float = 0.0245) -> float:
    """
    Linear motion law.

    Parameters
    ----------
    t : float
        Time (s).
    L0 : float, optional
        Initial value (default 1e-3).
    v : float, optional
        Constant velocity (default 0.0245).

    Returns
    -------
    float
        Position/length at time t.
    """
    return L0 + v * t

def quadratic(t: float, L0: float = 1e-3, c: float = 0.01225) -> float:
    """
    Quadratic motion law.

    Parameters
    ----------
    t : float
        Time (s).
    L0 : float, optional
        Initial value (default 1e-3).
    c : float, optional
        Quadratic coefficient (default 0.01225).

    Returns
    -------
    float
        Position/length at time t.
    """
    return L0 + c * t**2

def exponential(t: float, L0: float = 1e-3, A: float = 1e-3, k: float = 1.9455) -> float:
    """
    Exponential motion law.

    Parameters
    ----------
    t : float
        Time (s).
    L0 : float, optional
        Initial value (default 1e-3).
    A : float, optional
        Amplitude of the exponential term (default 1e-3).
    k : float, optional
        Exponential growth rate (default 1.9455).

    Returns
    -------
    float
        Position/length at time t.
    """
    return L0 + A * np.exp(k * t)
