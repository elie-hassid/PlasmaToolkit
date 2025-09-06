import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def pool_reactions(file, points_nbr=200, log_interp=True, eps=1e-30):
    """
    Reads an input csv file (typically generated from a spreadsheet software).
    Column i [0, 2, ..., n-1] is the energy and column j [1, 3, ..., n] is the cross-section.
    Each process to be pooled has to be put on a 2 columns i,j.
        1. Those processes are read.
        2. An energy grid is created with points_nbr cells.
        3. Cross sections for each process are interpolated (log-log by default) on the energy grid.
        4. Cross sections are summed and the energy and cross-section columns are returned.
    """
    data = pd.read_csv(file)
    n_cols = data.shape[1]

    E_list = [data.iloc[:, i].to_numpy(dtype=float) for i in range(0, n_cols, 2)]
    Sigma_list = [data.iloc[:, i].to_numpy(dtype=float) for i in range(1, n_cols, 2)]

    E_min = min([np.nanmin(E) for E in E_list])
    E_max = max([np.nanmax(E) for E in E_list])

    if log_interp:
        E_pool = np.logspace(np.log10(E_min), np.log10(E_max), points_nbr)
    else:
        E_pool = np.linspace(E_min, E_max, points_nbr)

    Sigma_pool = np.zeros_like(E_pool)
    for E, Sigma in zip(E_list, Sigma_list):
        # Ignorer les valeurs manquantes
        mask = ~np.isnan(E) & ~np.isnan(Sigma)
        E_clean = E[mask]
        Sigma_clean = Sigma[mask]

        if log_interp:
            Sigma_safe = np.maximum(Sigma_clean, eps)
            interp_func = interp1d(E_clean, np.log(Sigma_safe), kind='linear',
                                   bounds_error=False, fill_value=np.log(eps))
            Sigma_interp = np.exp(interp_func(E_pool))
        else:
            interp_func = interp1d(E_clean, Sigma_clean, kind='linear',
                                   bounds_error=False, fill_value=0.0)
            Sigma_interp = interp_func(E_pool)
        Sigma_pool += Sigma_interp

    return E_pool, Sigma_pool


if __name__ == "__main__":

    E_new, sigma_pooled = pool_reactions("n2vib.csv", points_nbr=100, log_interp=True)
    print("Energy (eV)    Pooled cross-section (m^2)")
    print("-----------------------------------------")
    for E, sigma in zip(E_new, sigma_pooled):
        print(f"{E:.5E}    {sigma:.5E}")
