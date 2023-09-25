""" This module contains the numba optimised functions for the energy terms, change in energy calculations and implementions 
for specific cases encountered during my thesis. The functions are called in the numba compiled total energy function in mcpy\Energies\numba_energies.py"""


from numba import njit, prange
import numpy as np


# Global constants
M_0 = 4*np.pi*1e-7  # Magnetic permeability of free space


@njit
def zeeman_energy(grid: np.ndarray, zeeman_H: np.ndarray, Ms: np.float64, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """ Calculate the Zeeman energy of the system

    Args:
        grid (np.array): 3D vector field
        zeeman_H (np.array): Zeeman field
        Ms (np.float64): Saturation magnetisation
        dx (np.float64): x discretisation of the grid
        dy (np.float64): y discretisation of the grid
        dz (np.float64): z discretisation of the grid

    Returns:
        np.float64: Zeeman energy of the system
    """
    energy = - M_0 * Ms * \
        np.sum(grid[1:-1, 1:-1, 1:-1] * zeeman_H) * dx * dy * dz
    return energy


@njit
def anisotropy_energy(grid: np.ndarray, anisotropy_K: np.float64, anisotropy_u: np.ndarray, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """ Calculate the anisotropy energy of the system

    Args:
        grid (np.array): 3D vector field
        anisotropy_K (np.float64): Anisotropy constant
        anisotropy_u (np.array): Anisotropy axis
        dx (np.float64): x discretisation of the grid
        dy (np.float64): y discretisation of the grid
        dz (np.float64): z discretisation of the grid

    Returns:
        np.float64: Anisotropy energy of the system
    """

    energy = np.cross(grid[1:-1, 1:-1, 1:-1], anisotropy_u)
    # total energy of the system
    energy = anisotropy_K*np.sum(energy**2)*dx*dy*dz

    return energy


@njit(fastmath=True)
def exchange_energy(grid: np.ndarray, exchange_A: np.float64, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """ Calculate the exchange energy of the system

    Args:
        grid (np.array): 3D vector field
        exchange_A (np.float64): Exchange constant
        dx (np.float64): x discretisation of the grid
        dy (np.float64): y discretisation of the grid
        dz (np.float64): z discretisation of the grid

    Returns:
        np.float64: Exchange energy of the system

    >>> grid = np.ones((5, 5, 5, 3))
    >>> exchange_energy(grid, 1, 1, 1, 1)
    -96.0
    """

    laplacian_M = (((grid[2:, 1:-1, 1:-1] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[:-2, 1:-1, 1:-1])/dx**2) +
                   ((grid[1:-1, 2:, 1:-1] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[1:-1, :-2, 1:-1])/dy**2) +
                   ((grid[1:-1, 1:-1, 2:] - 2 * grid[1:-1,
                     1:-1, 1:-1] + grid[1:-1, 1:-1, :-2])/dz**2)
                   )

    # dot product of m and laplacian_M
    energy = np.sum(grid[1:-1, 1:-1, 1:-1]*laplacian_M)
    energy = -exchange_A*energy*dx*dy*dz  # total energy of the system
    return energy


@njit
def dmi_Cnvz(grid: np.ndarray, D: np.ndarray, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """ Calculate the DMI energy of the system. Uses 'Cnv_z' point group for DMI energy calculation.

    Args:
        grid (np.array): 3D vector field
        D (np.array): DMI constant grid
        dx (np.float64): x discretisation of the grid
        dy (np.float64): y discretisation of the grid
        dz (np.float64): z discretisation of the grid

    Returns:
        np.float64: DMI energy of the system
    """

    gradM_z = np.empty_like(grid[1:-1, 1:-1, 1:-1], dtype='float64')
    gradM_z[..., 0] = (grid[2:, 1:-1, 1:-1, 2] -
                       grid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
    gradM_z[..., 1] = (grid[1:-1, 2:, 1:-1, 2] -
                       grid[1:-1, :-2, 1:-1, 2]) / (2 * dy)
    gradM_z[..., 2] = (grid[1:-1, 1:-1, 2:, 2] -
                       grid[1:-1, 1:-1, :-2, 2]) / (2 * dz)

    # divergence of m vector
    div_M = (
        (grid[2:, 1:-1, 1:-1, 0] - grid[:-2, 1:-1, 1:-1, 0]) / (2 * dx) +
        (grid[1:-1, 2:, 1:-1, 1] - grid[1:-1, :-2, 1:-1, 1]) / (2 * dy) +
        (grid[1:-1, 1:-1, 2:, 2] - grid[1:-1, 1:-1, :-2, 2]) / (2 * dz)
    )

    # dot product of m and gradient of z component of m vector
    m_del_mz = np.sum(grid[1:-1, 1:-1, 1:-1]*gradM_z, axis=-1)
    mz_div_m = grid[1:-1, 1:-1, 1:-1, 2]*div_M  # mz∇⋅m

    energy = np.sum(D*(m_del_mz - mz_div_m))*dx * \
        dy*dz  # total energy of the system

    return energy


@njit(fastmath=True)
def dmi_D2d_z(grid: np.ndarray, D: np.ndarray, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """ Calculate the DMI energy of the system. Uses 'D2d_z' point group for DMI energy calculation.

    Args:
        grid (np.array): 3D vector field
        D (np.array): DMI constant grid
        dx (np.float64): x discretisation of the grid
        dy (np.float64): y discretisation of the grid
        dz (np.float64): z discretisation of the grid

    Returns:
        np.float64: DMI energy of the system
    """

    gradM_x = np.empty_like(grid[1:-1, 1:-1, 1:-1], dtype='float64')
    gradM_x[..., 0] = (grid[2:, 1:-1, 1:-1, 0] -
                       grid[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
    gradM_x[..., 1] = (grid[1:-1, 2:, 1:-1, 1] -
                       grid[1:-1, :-2, 1:-1, 1]) / (2 * dy)
    gradM_x[..., 2] = (grid[1:-1, 1:-1, 2:, 2] -
                       grid[1:-1, 1:-1, :-2, 2]) / (2 * dz)

    gradM_y = np.empty_like(grid[1:-1, 1:-1, 1:-1], dtype='float64')
    gradM_y[..., 0] = (grid[2:, 1:-1, 1:-1, 0] -
                       grid[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
    gradM_y[..., 1] = (grid[1:-1, 2:, 1:-1, 1] -
                       grid[1:-1, :-2, 1:-1, 1]) / (2 * dy)
    gradM_y[..., 2] = (grid[1:-1, 1:-1, 2:, 2] -
                       grid[1:-1, 1:-1, :-2, 1]) / (2 * dz)

    # dm/dx x ^x - dm/dy x ^y
    cross_x = np.cross(gradM_x, np.array([1, 0, 0], dtype='float64'))
    cross_y = np.cross(gradM_y, np.array([0, 1, 0], dtype='float64'))
    res = cross_x - cross_y

    # Total energy
    energy = np.sum(D*grid[1:-1, 1:-1, 1:-1]*res)*dx * \
        dy*dz  # total energy of the system
    return energy


@njit(fastmath=True)
def dmi_TO(grid: np.ndarray, D: np.ndarray, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """ Calculate the DMI energy of the system. Uses 'T or O' point group for DMI energy calculation.

    Args:
        grid (np.array): 3D vector field
        D (np.array): DMI constant grid
        dx (np.float64): x discretisation of the grid
        dy (np.float64): y discretisation of the grid
        dz (np.float64): z discretisation of the grid

    Returns:
        np.float64: DMI energy of the system
    """

    curl = np.empty_like(grid[1:-1, 1:-1, 1:-1], dtype='float64')
    curl[..., 0] = (grid[1:-1, 2:, 1:-1, 2] - grid[1:-1, :-2, 1:-1, 2]) / (2 * dy) - \
        (grid[1:-1, 1:-1, 2:, 1] - grid[1:-1, 1:-1, :-2, 1]) / (2 * dz)

    curl[..., 1] = (grid[1:-1, 1:-1, 2:, 0] - grid[1:-1, 1:-1, :-2, 0]) / (2 * dz) - \
        (grid[2:, 1:-1, 1:-1, 2] - grid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)

    curl[..., 2] = (grid[2:, 1:-1, 1:-1, 1] - grid[:-2, 1:-1, 1:-1, 1]) / (2 * dx) - \
        (grid[1:-1, 2:, 1:-1, 0] - grid[1:-1, :-2, 1:-1, 0]) / (2 * dy)

    # dot product of m and curl
    energy = np.sum(grid[1:-1, 1:-1, 1:-1]*curl, axis=-1)
    energy = np.sum(D*energy)*dx*dy*dz  # total energy of the system
    return energy


@njit(fastmath=True)
def numba_delta(grid_ex: np.ndarray, grid_dmi: np.ndarray, spins: np.ndarray, Ms: np.float64, zeeman_H: np.ndarray, exchange_A: np.float64, anisotropy_K: np.float64, anisotropy_u: np.ndarray, dmi_D: np.ndarray, Dtype: str, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """ Calculates the change in energy between the current state and the proposed state. Uses numba compiled functions for energy calculation.

    Args:
        grid_ex (np.ndarray): 5x5x5 grid for exchange with neumann boundary conditions
        grid_dmi (np.ndarray): 5x5x5 grid for DMI with dirichlet boundary conditions
        spins (np.ndarray): (Previous and proposed) spin configuration
        Ms (float): Saturation magnetisation
        zeeman_H (np.ndarray): Zeeman field
        exchange_A (np.float64): Exchange constant
        anisotropy_K (np.float64): Anisotropy constant
        anisotropy_u (np.ndarray): Anisotropy axis
        dmi_D (np.ndarray): DMI constant
        Dtype (str): DMI type or Crystal class
        dx (np.float64): x discretisation of the grid
        dy (np.float64): y discretisation of the grid
        dz (np.float64): z discretisation of the grid

    Returns:
        np.float64: Change in energy between the current state and the proposed state
    """

    delta = np.array([0, 0], dtype='float64')

    for i in prange(spins.shape[0]):  # TODO: Parallelize this loop
        grid_ex[2, 2, 2] = spins[i]
        grid_dmi[2, 2, 2] = spins[i]

        energy = 0.0

        if zeeman_H is not None:
            energy += zeeman_energy(grid_dmi, zeeman_H, Ms, dx, dy, dz)
        if exchange_A is not None:
            energy += exchange_energy(grid_ex, exchange_A, dx, dy, dz)
        if anisotropy_K is not None:
            energy += anisotropy_energy(grid_dmi,
                                        anisotropy_K, anisotropy_u, dx, dy, dz)
        if Dtype is not None:
            if Dtype == 'Cnv_z':
                energy += dmi_Cnvz(grid_dmi, dmi_D, dx, dy, dz)
            elif Dtype == 'T':
                energy += dmi_TO(grid_dmi, dmi_D, dx, dy, dz)
            elif Dtype == 'D2d_z':
                energy += dmi_D2d_z(grid_dmi, dmi_D, dx, dy, dz)
            else:
                raise ValueError(
                    f'Dtype must be either Cnvz, T or D2d_z. Got: {Dtype}')
        delta[i] = energy

    return delta[1] - delta[0]


"""Below functions are created for specific cases for my thesis. By reducing function calls and decision statements to speed up the code."""


@njit(fastmath=True)
def delta_energy(grid: np.ndarray, spins: np.ndarray, Ms: np.float64, zeeman_H: np.ndarray, exchange_A: np.float64, anisotropy_K: np.float64, anisotropy_u: np.ndarray, dmi_D: np.ndarray, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """ Calculate the energy difference between the current state and the proposed state. Uses 'Conv_z' point group for DMI energy calculation.

    Args:
        grid (np.array): 3D vector field
        spins (np.array): (Previous and proposed) spin configuration
        Ms (float): Saturation magnetisation
        zeeman_H (np.array): Zeeman field
        exchange_A (np.float64): Exchange constant
        anisotropy_K (np.float64): Anisotropy constant
        anisotropy_u (np.array): Anisotropy axis
        dmi_D (np.array): DMI constant
        dx (np.float64): x discretisation of the grid
        dy (np.float64): y discretisation of the grid
        dz (np.float64): z discretisation of the grid

    Returns:
        np.float64: Energy difference between the current state and the proposed state
    """

    delta = np.array([0, 0], dtype='float64')

    for i in prange(spins.shape[0]):
        grid[2, 2, 2] = spins[i]

        zeeman = -M_0 * Ms * \
            np.sum(grid[1:-1, 1:-1, 1:-1] * zeeman_H) * dx * dy * dz

        # aniso = np.cross(grid[1:-1, 1:-1, 1:-1], anisotropic_u)
        # aniso = anisotropic_K*np.sum(aniso**2)*dx*dy*dz

        laplacian_M = (((grid[2:, 1:-1, 1:-1] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[:-2, 1:-1, 1:-1])/dx**2) +
                       ((grid[1:-1, 2:, 1:-1] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[1:-1, :-2, 1:-1])/dy**2) +
                       ((grid[1:-1, 1:-1, 2:] - 2 * grid[1:-1,
                         1:-1, 1:-1] + grid[1:-1, 1:-1, :-2])/dz**2)
                       )
        # dot product of m and laplacian_M
        exchange = np.sum(grid[1:-1, 1:-1, 1:-1]*laplacian_M)
        exchange = -exchange_A*exchange*dx*dy*dz  # total energy of the system

        gradM_z = np.empty_like(grid[1:-1, 1:-1, 1:-1], dtype='float64')
        gradM_z[..., 0] = (grid[2:, 1:-1, 1:-1, 2] -
                           grid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
        gradM_z[..., 1] = (grid[1:-1, 2:, 1:-1, 2] -
                           grid[1:-1, :-2, 1:-1, 2]) / (2 * dy)
        gradM_z[..., 2] = (grid[1:-1, 1:-1, 2:, 2] -
                           grid[1:-1, 1:-1, :-2, 2]) / (2 * dz)

        # divergence of m vector
        div_M = ((grid[2:, 1:-1, 1:-1, 0] - grid[:-2, 1:-1, 1:-1, 0]) / (2 * dx) +
                 (grid[1:-1, 2:, 1:-1, 1] - grid[1:-1, :-2, 1:-1, 1]) / (2 * dy) +
                 (grid[1:-1, 1:-1, 2:, 2] - grid[1:-1, 1:-1, :-2, 2]) / (2 * dz)
                 )
        # dot product of m and gradient of z component of m vector
        m_del_mz = np.sum(dmi_D*grid[1:-1, 1:-1, 1:-1]*gradM_z)
        mz_div_m = np.sum(dmi_D*grid[1:-1, 1:-1, 1:-1, 2]*div_M)  # mz∇⋅m

        dmi = (m_del_mz - mz_div_m) * dx * dy * \
            dz  # Total energy of the system

        delta[i] = zeeman + exchange + dmi  # + aniso

    return delta[1] - delta[0]


@njit(fastmath=True)
def delta_energy2(grid_ex: np.ndarray, grid_dmi: np.ndarray, spins: np.ndarray, Ms: np.float64, zeeman_H: np.ndarray, exchange_A: np.float64, anisotropy_K: np.float64, anisotropy_u: np.ndarray, dmi_D: np.ndarray, dx: np.float64, dy: np.float64, dz: np.float64) -> np.float64:
    """ Calculate the energy difference between the current state and the proposed state. Uses 'T or O' point group for DMI energy calculation.
        For Bloch point simulations

    Args:
        grid_ex (np.ndarray): 5x5x5 grid for exchange with neumann boundary conditions
        grid_dmi (np.ndarray): 5x5x5 grid for DMI with dirichlet boundary conditions
        spins (np.ndarray): (Previous and proposed) spin configuration
        Ms (np.float64): Saturation magnetisation
        zeeman_H (np.ndarray): Zeeman field
        exchange_A (np.float64): Exchange constant
        anisotropy_K (np.float64): Anisotropy constant
        anisotropy_u (np.ndarray): Anisotropy axis
        dmi_D (np.ndarray): DMI constant
        dx (np.float64): x discretisation of the grid
        dy (np.float64): y discretisation of the grid
        dz (np.float64): z discretisation of the grid

        Returns:
            np.float64: Energy difference between the current state and the proposed state
        """

    delta = np.array([0, 0], dtype='float64')

    for i in prange(spins.shape[0]):
        grid_ex[2, 2, 2] = spins[i]
        grid_dmi[2, 2, 2] = spins[i]

        zeeman = - M_0 * Ms * \
            np.sum(grid_dmi[1:-1, 1:-1, 1:-1] * zeeman_H) * dx * dy * dz

        laplacian_M = (((grid_ex[2:, 1:-1, 1:-1] - 2 * grid_ex[1:-1, 1:-1, 1:-1] + grid_ex[:-2, 1:-1, 1:-1])/dx**2) +
                       ((grid_ex[1:-1, 2:, 1:-1] - 2 * grid_ex[1:-1, 1:-1, 1:-1] + grid_ex[1:-1, :-2, 1:-1])/dy**2) +
                       ((grid_ex[1:-1, 1:-1, 2:] - 2 * grid_ex[1:-1,
                         1:-1, 1:-1] + grid_ex[1:-1, 1:-1, :-2])/dz**2)
                       )
        # dot product of m and laplacian_M
        exchange = np.sum(grid_ex[1:-1, 1:-1, 1:-1]*laplacian_M)
        exchange = -exchange_A*exchange*dx*dy*dz  # total energy of the system

        curl = np.empty_like(grid_dmi[1:-1, 1:-1, 1:-1], dtype='float64')
        curl[..., 0] = (grid_dmi[1:-1, 2:, 1:-1, 2] - grid_dmi[1:-1, :-2, 1:-1, 2]) / (2 * dy) - \
            (grid_dmi[1:-1, 1:-1, 2:, 1] -
             grid_dmi[1:-1, 1:-1, :-2, 1]) / (2 * dz)

        curl[..., 1] = (grid_dmi[1:-1, 1:-1, 2:, 0] - grid_dmi[1:-1, 1:-1, :-2, 0]) / (2 * dz) - \
            (grid_dmi[2:, 1:-1, 1:-1, 2] -
             grid_dmi[:-2, 1:-1, 1:-1, 2]) / (2 * dx)

        curl[..., 2] = (grid_dmi[2:, 1:-1, 1:-1, 1] - grid_dmi[:-2, 1:-1, 1:-1, 1]) / (2 * dx) - \
            (grid_dmi[1:-1, 2:, 1:-1, 0] -
             grid_dmi[1:-1, :-2, 1:-1, 0]) / (2 * dy)

        # dot product of m and curl and summing over all the elements
        dmi = np.sum(grid_dmi[1:-1, 1:-1, 1:-1]*curl, axis=-1)

        dmi = np.sum(dmi_D * dmi) * dx * dy * dz

        delta[i] = exchange + dmi + zeeman

    return delta[1] - delta[0]
