import numpy as np
# from mcpy.system import Grid

""" This module contains numpy functions for the energy terms and energy change calculations. """

# Global constants
M_0 = 4*np.pi*1e-7  # Magnetic permeability of free space


def zeeman_energy(grid: np.ndarray, zeeman_H: np.ndarray, Ms: np.float64, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """Calculates the Zeeman energy of the system

    Args:
        grid (np.ndarray): 3D array of spins
        zeeman_H (np.ndarray): External magnetic field
        Ms (np.float64): Saturation magnetisation
        dx (np.float64): Grid spacing in x direction
        dy (np.float64): Grid spacing in y direction
        dz (np.float64): Grid spacing in z direction

    Returns:
        energy (np.float64): Zeeman energy of the system

    >>> grid = np.ones((5, 5, 5, 3))
    >>> zeeman_H = np.array([0, 0, 1])
    >>> Ms = 10
    >>> dx, dy, dz = np.array([1, 1, 1])
    >>> zeeman_energy(grid, zeeman_H, Ms, dx, dy, dz)
    -500.0
    """
    if grid.shape == (5, 5, 5, 3):
        grid = grid[1:-1, 1:-1, 1:-1]

    energy = -M_0 * Ms * np.sum(grid * zeeman_H) * dx * dy * dz

    return energy


def anisotropy_energy(grid: np.ndarray, anisotropy_K: np.float64, anisotropy_u: np.ndarray, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """Calculates the anisotropy energy of the system

    Args:
        grid (np.ndarray): 3D array of spins
        anisotropy_K (np.float64): Anisotropy constant
        anisotropy_u (np.ndarray): Anisotropy axis
        dx (np.float64): Grid spacing in x direction
        dy (np.float64): Grid spacing in y direction
        dz (np.float64): Grid spacing in z direction

    Returns:
        energy (np.float64): Anisotropy energy of the system

    >>> grid = np.ones((5, 5, 5, 3))
    >>> anisotropy_K = 1
    >>> anisotropy_u = np.array([0, 0, 1])
    >>> dx, dy, dz = np.array([1, 1, 1])
    >>> anisotropy_energy(grid, anisotropy_K, anisotropy_u, dx, dy, dz)
    0.0
    """

    if grid.shape == (5, 5, 5, 3):
        grid = grid[1:-1, 1:-1, 1:-1]
    energy = np.cross(grid, anisotropy_u)
    energy = anisotropy_K * np.sum(energy**2) * dx * dy * dz

    return energy


def exchange_energy(grid: np.ndarray, exchange_A: np.float64, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """Calculates the exchange energy of the system

    Args:
        grid (np.ndarray): 3D array of spins
        exchange_A (np.float64): Exchange constant
        dx (np.float64): Grid spacing in x direction
        dy (np.float64): Grid spacing in y direction
        dz (np.float64): Grid spacing in z direction

    Returns:
        energy (np.float64): Exchange energy of the system

    >>> grid = np.ones((5, 5, 5, 3))
    >>> exchange_A = 1
    >>> dx, dy, dz = np.array([1, 1, 1])
    >>> exchange_energy(grid, exchange_A, dx, dy, dz)
    -500.0
    """

    if grid.shape == (5, 5, 5, 3):
        pgrid = grid
    else:
        pass
        pgrid = np.pad(grid, ((1, 1), (1, 1), (1, 1), (0, 0)),
                       mode='edge')  # Padded grid with Neumann boundary conditions

    laplacian_M = (((pgrid[2:, 1:-1, 1:-1] - 2 * pgrid[1:-1, 1:-1, 1:-1] + pgrid[:-2, 1:-1, 1:-1]) / dx**2) +
                   ((pgrid[1:-1, 2:, 1:-1] - 2 * pgrid[1:-1, 1:-1, 1:-1] + pgrid[1:-1, :-2, 1:-1]) / dy**2) +
                   ((pgrid[1:-1, 1:-1, 2:] - 2 * pgrid[1:-1, 1:-1, 1:-1] + pgrid[1:-1, 1:-1, :-2]) / dz**2))

    energy = np.sum(pgrid[1:-1, 1:-1, 1:-1] * laplacian_M)
    energy = -exchange_A * energy * dx * dy * dz

    return energy


def dmi_energy(grid: np.ndarray, Dtype: str, D: np.ndarray, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """Calculates the DMI energy of the system

    Args:
        grid (np.ndarray): 3D array of spins
        Dtype (str): DMI type or Crystal class
        D (np.ndarray): DMI constant for the segmented 5x5x5 grid
        dx (np.float64): Grid spacing in x direction
        dy (np.float64): Grid spacing in y direction
        dz (np.float64): Grid spacing in z direction

    Returns:
        energy (np.float64): DMI energy of the system

    >>> grid = np.ones((5, 5, 5, 3))
    >>> Dtype = 'D2d_z'
    >>> D = 1
    >>> dx, dy, dz = np.array([1, 1, 1])
    >>> dmi_energy(grid, Dtype, D, dx, dy, dz)
    0.0 
    """
    if grid.shape == (5, 5, 5, 3):
        pgrid = grid
    else:
        pgrid = np.pad(grid, ((1, 1), (1, 1), (1, 1), (0, 0)),
                       mode='constant', constant_values=0)  # Padded grid with Dirichlet boundary conditions
        D = D[1:-1, 1:-1, 1:-1]

    if Dtype == 'Cnv_z':
        # calculate gradient of z component of m vector with respect to each axis
        gradM_z = np.empty_like(pgrid[1:-1, 1:-1, 1:-1], dtype='float64')
        gradM_z[..., 0] = (pgrid[2:, 1:-1, 1:-1, 2] -
                           pgrid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
        gradM_z[..., 1] = (pgrid[1:-1, 2:, 1:-1, 2] -
                           pgrid[1:-1, :-2, 1:-1, 2]) / (2 * dy)
        gradM_z[..., 2] = (pgrid[1:-1, 1:-1, 2:, 2] -
                           pgrid[1:-1, 1:-1, :-2, 2]) / (2 * dz)

        # divergence of m vector
        div_M = ((pgrid[2:, 1:-1, 1:-1, 0] - pgrid[:-2, 1:-1, 1:-1, 0]) / (2 * dx) +
                 (pgrid[1:-1, 2:, 1:-1, 1] - pgrid[1:-1, :-2, 1:-1, 1]) / (2 * dy) +
                 (pgrid[1:-1, 1:-1, 2:, 2] -
                  pgrid[1:-1, 1:-1, :-2, 2]) / (2 * dz)
                 )

        # dot product of m and gradient of z component of m vector
        m_del_mz = np.sum(pgrid[1:-1, 1:-1, 1:-1]*gradM_z, axis=-1)
        mz_div_m = pgrid[1:-1, 1:-1, 1:-1, 2]*div_M  # mz∇⋅m

        energy = np.sum(D*(m_del_mz - mz_div_m))*dx * \
            dy*dz  # total energy of the system

    elif Dtype == 'D2d_z':

        gradM_x = np.empty_like(pgrid[1:-1, 1:-1, 1:-1], dtype='float64')
        gradM_x[..., 0] = (pgrid[2:, 1:-1, 1:-1, 0] -
                           pgrid[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
        gradM_x[..., 1] = (pgrid[1:-1, 2:, 1:-1, 1] -
                           pgrid[1:-1, :-2, 1:-1, 1]) / (2 * dy)
        gradM_x[..., 2] = (pgrid[1:-1, 1:-1, 2:, 2] -
                           pgrid[1:-1, 1:-1, :-2, 2]) / (2 * dz)

        gradM_y = np.empty_like(pgrid[1:-1, 1:-1, 1:-1], dtype='float64')
        gradM_y[..., 0] = (pgrid[2:, 1:-1, 1:-1, 0] -
                           pgrid[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
        gradM_y[..., 1] = (pgrid[1:-1, 2:, 1:-1, 1] -
                           pgrid[1:-1, :-2, 1:-1, 1]) / (2 * dy)
        gradM_y[..., 2] = (pgrid[1:-1, 1:-1, 2:, 2] -
                           pgrid[1:-1, 1:-1, :-2, 1]) / (2 * dz)

        # dm/dx x ^x - dm/dy x ^y
        cross_x = np.cross(gradM_x, np.array([1, 0, 0], dtype='float64'))
        cross_y = np.cross(gradM_y, np.array([0, 1, 0], dtype='float64'))
        res = cross_x - cross_y

        # Total energy
        energy = np.sum(D*pgrid[1:-1, 1:-1, 1:-1]*res)*dx * \
            dy*dz  # total energy of the system

    elif type == 'Cnv_xy':
        # TODO: implement Cnv_xy
        raise NotImplementedError("Cnv_xy is not implemented yet")

    else:

        curl = np.empty_like(pgrid[1:-1, 1:-1, 1:-1], dtype='float64')
        curl[..., 0] = (pgrid[1:-1, 2:, 1:-1, 2] - pgrid[1:-1, :-2, 1:-1, 2]) / (2 * dy) - \
            (pgrid[1:-1, 1:-1, 2:, 1] - pgrid[1:-1, 1:-1, :-2, 1]) / (2 * dz)

        curl[..., 1] = (pgrid[1:-1, 1:-1, 2:, 0] - pgrid[1:-1, 1:-1, :-2, 0]) / (2 * dz) - \
            (pgrid[2:, 1:-1, 1:-1, 2] - pgrid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)

        curl[..., 2] = (pgrid[2:, 1:-1, 1:-1, 1] - pgrid[:-2, 1:-1, 1:-1, 1]) / (2 * dx) - \
            (pgrid[1:-1, 2:, 1:-1, 0] - pgrid[1:-1, :-2, 1:-1, 0]) / (2 * dy)

        # dot product of m and curl
        energy = np.sum(pgrid[1:-1, 1:-1, 1:-1]*curl, axis=-1)
        energy = np.sum(D*energy)*dx * \
            dy*dz  # total energy of the system

    return energy


def numpy_delta(grid, spins: np.ndarray, grid_ex: np.ndarray, grid_dmi: np.ndarray, D: np.ndarray, zeeman_H: np.ndarray) -> np.float64:
    """Calculates the change in energy of the system using numpy

    Args:
        grid (Grid): Grid object
        spins (np.ndarray): Array of spins
        grid_ex (np.ndarray): Exchange grid
        grid_dmi (np.ndarray): DMI grid

    Returns:
        delta (np.float64): Change in energy of the system


    >>> grid = np.ones((5, 5, 5, 3))
    >>> spins = np.array([[1, 0, 0], [0, 1, 0]])
    >>> grid_ex = np.ones((5, 5, 5, 3))
    >>> grid_dmi = np.ones((5, 5, 5, 3))
    >>> D = 1
    >>> zeeman_H = np.array([0, 0, 1])
    >>> numpy_delta(grid, spins, grid_ex, grid_dmi, D, zeeman_H)
    0.0
    """

    delta = np.array([0, 0], dtype='float64')

    for i in range(spins.shape[0]):
        energy = 0.0
        grid_dmi[2, 2, 2] = spins[i]
        grid_ex[2, 2, 2] = spins[i]

        if zeeman_H is not None:
            energy += zeeman_energy(grid_ex, zeeman_H,
                                    grid.Ms, grid.dx, grid.dy, grid.dz)

        if grid.anisotropy_K is not None:
            energy += anisotropy_energy(grid_ex, grid.anisotropy_K,
                                        grid.anisotropy_u, grid.dx, grid.dy, grid.dz)
        if grid.exchange_A is not None:
            energy += exchange_energy(grid_ex, grid.exchange_A,
                                      grid.dx, grid.dy, grid.dz)
        if grid.dmi_D is not None:
            energy += dmi_energy(grid_dmi, grid.Dtype, D,
                                 grid.dx, grid.dy, grid.dz)

        delta[i] = energy

    return delta[1] - delta[0]
