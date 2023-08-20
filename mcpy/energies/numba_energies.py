from numba import njit, prange
import numpy as np

""" This module contains the numba optimised functions for the energy terms, change in energy calculations and implementions 
for specific cases encountered during my thesis. The functions are called in the numba compiled total energy function in mcpy\Energies\numba_energies.py"""


# Global constants
M_0 = 4*np.pi*1e-7  # Magnetic permeability of free space


@njit
def zeeman_energy(grid, zeeman_H, Ms, dx, dy, dz):
    """ Calculate the Zeeman energy of the system

    Args:
        grid (np.array): 3D vector field
        zeeman_H (np.array): Zeeman field
        Ms (float): Saturation magnetisation
        dx (float): x dimension of the grid
        dy (float): y dimension of the grid
        dz (float): z dimension of the grid

    Returns:
        float: Zeeman energy of the system
    """
    energy = - M_0 * Ms * \
        np.sum(grid[1:-1, 1:-1, 1:-1] * zeeman_H) * dx * dy * dz
    return energy


@njit
def anisotropy_energy(grid, anisotropy_K, anisotropy_u, dx, dy, dz):
    """ Calculate the anisotropy energy of the system

    Args:
        grid (np.array): 3D vector field
        anisotropy_K (float): Anisotropy constant
        anisotropy_u (np.array): Anisotropy axis
        dx (float): x dimension of the grid
        dy (float): y dimension of the grid
        dz (float): z dimension of the grid

    Returns:
        float: Anisotropy energy of the system
    """

    energy = np.cross(grid[1:-1, 1:-1, 1:-1], anisotropy_u)
    # total energy of the system
    energy = anisotropy_K*np.sum(energy**2)*dx*dy*dz

    return energy


@njit(fastmath=True)
def exchange_energy(grid, exchange_A, dx, dy, dz):
    """ Calculate the exchange energy of the system

    Args:
        grid (np.array): 3D vector field with neumann boundary conditions
        exchange_A (float): Exchange constant
        dx (float): x dimension of the grid
        dy (float): y dimension of the grid
        dz (float): z dimension of the grid

    Returns:
        float: Exchange energy of the system

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
def dmi_Cnvz(grid, D, dx, dy, dz):
    """ Calculate the DMI energy of the system. Uses 'Cnv_z' point group for DMI energy calculation.

    Args:
        grid (np.array): 3D vector field
        D (np.array): DMI constant grid
        dx (float): x dimension of the grid
        dy (float): y dimension of the grid
        dz (float): z dimension of the grid

    Returns:
        float: DMI energy of the system
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
def dmi_D2d_z(grid, D, dx, dy, dz):

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
def dmi_TO(grid, D, dx, dy, dz):
    """ Calculate the DMI energy of the system. Uses 'T or O' point group for DMI energy calculation.

    Args:
        grid (np.array): 3D vector field
        D (np.array): DMI constant grid
        dx (float): x dimension of the grid
        dy (float): y dimension of the grid
        dz (float): z dimension of the grid

    Returns:
        float: DMI energy of the system
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
def numba_delta(grid_ex, grid_dmi, spins, Ms, zeeman_H, exchange_A, anisotropy_K, anisotropy_u, D, Dtype, dx, dy, dz):
    """ Calculates the change in energy between the current state and the proposed state. Uses numba compiled functions for energy calculation.

    Args:
        grid_ex (np.array): 3D vector field with neumann boundary conditions
        grid_dmi (np.array): 3D vector field with dirichlet boundary conditions
        spins (np.array): (Previous and proposed) spin configuration
        Ms (float): Saturation magnetisation
        zeeman_H (np.array): Zeeman field
        exchange_A (float): Exchange constant
        anisotropy_K (float): Anisotropy constant
        anisotropy_u (np.array): Anisotropy axis
        D (float): DMI constant
        Dtype (np.dtype): DMI type or Crystal class
        dx (float): x dimension of the grid
        dy (float): y dimension of the grid
        dz (float): z dimension of the grid

    Returns:
        float: Energy difference between the current state and the proposed state
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
                energy += dmi_Cnvz(grid_dmi, D, dx, dy, dz)
            elif Dtype == 'T':
                energy += dmi_TO(grid_dmi, D, dx, dy, dz)
            elif Dtype == 'D2d_z':
                energy += dmi_D2d_z(grid_dmi, D, dx, dy, dz)
            else:
                raise ValueError(
                    f'Dtype must be either Cnvz, T or D2d_z. Got: {Dtype}')
        delta[i] = energy

    return delta[1] - delta[0]


"""Below functions are created for specific cases for my thesis. By reducing function calls and decision statements to speed up the code."""


@njit(fastmath=True)
def delta_energy(grid, spins, Ms, zeeman_H, exchange_A, anisotropy_K, anisotropy_u, dmi_D, dx, dy, dz):
    """ Calculate the energy difference between the current state and the proposed state. Uses 'Conv_z' point group for DMI energy calculation.

    Args:
        grid (np.array): 3D vector field
        spins (np.array): (Previous and proposed) spin configuration
        m0 (float): Magnitude of the spin
        Ms (float): Saturation magnetisation
        zeeman_H (np.array): Zeeman field
        exchange_A (float): Exchange constant
        anisotropy_K (float): Anisotropy constant
        anisotropy_u (np.array): Anisotropy axis
        dmi_D (float): DMI constant
        dx (float): x dimension of the grid
        dy (float): y dimension of the grid
        dz (float): z dimension of the grid

    Returns:
        float: Energy difference between the current state and the proposed state
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
def delta_energy2(grid_ex, grid_dmi, spins, Ms, zeeman_H, exchange_A, anisotropy_K, anisotropy_u, dmi_D, dx, dy, dz):
    """ Calculate the energy difference between the current state and the proposed state. Uses 'T or O' point group for DMI energy calculation.
        For Bloch point simulations
        Args:
            grid (np.array): 3D vector field
            spins (np.array): (Previous and proposed) spin configuration
            m0 (float): Magnitude of the spin
            Ms (float): Saturation magnetisation
            zeeman_H (np.array): Zeeman field
            exchange_A (float): Exchange constant
            dmi_D (float): DMI constant
            dx (float): x dimension of the grid
            dy (float): y dimension of the grid
            dz (float): z dimension of the grid

        Returns:
            float: Energy difference between the current state and the proposed state
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
