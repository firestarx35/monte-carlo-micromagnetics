
"""This module contains the main functions for implementing the Metropolis-Hastings Monte Carlo simulations. 
By changing the spin each iteartion by random angle with uniform distribution."""

import numpy as np
from numba import njit, prange
from tqdm import tqdm
from mcpy.energies.numpy_energies import numpy_delta

# Global constants

K_B = 1.38064852e-23  # Boltzmann constant


def driver_numpy(N: int, grid, zeeman_H: np.ndarray, temperature: np.float64) -> np.ndarray:
    """ Monte Carlo driver function for Numpy implementation

    Args:
        N (int): Number of Monte Carlo steps
        grid (mcpy.system.Grid): Grid object
        zeeman_H (np.ndarray): Zeeman field
        temperature (np.float64): Temperature in Kelvin

    Returns:
        grid (np.ndarray): Relaxed system
    """
    spins = np.zeros(
        (2, 3), dtype='float64')  # array to store original and proposed spin
    shape = grid.grid.shape

    if grid.dmi_D is None:
        D = None

    for _ in tqdm(range(N)):
        # 1. Randomly select a cell
        cell_x, cell_y, cell_z = np.random.randint(0, shape[:3])

        # if the cell is empty, select another cell
        while np.all(grid.grid[cell_x, cell_y, cell_z] == 0):
            cell_x, cell_y, cell_z = np.random.randint(0, shape[:3])

        # Segmenting the grid into 5x5x5 cells
        if 2 <= cell_x <= shape[0] - 3 and 2 <= cell_y <= shape[1] - 3 and 2 <= cell_z <= shape[2] - 3:
            grid_ex = grid.grid[cell_x-2:cell_x+3, cell_y -
                                2:cell_y+3, cell_z-2:cell_z+3].copy()
            grid_dmi = grid_ex

        else:
            # 5x5x5 grid for exchange with neumann boundary conditions
            grid_ex = np.empty((5, 5, 5, 3))
            # 5x5x5 grid for DMI with dirichlet boundary conditions
            grid_dmi = np.empty((5, 5, 5, 3))

            # this is done to avoid changing the indexes (cell_x, cell_y, cell_z) in the main grid
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        x, y, z = cell_x + i, cell_y + j, cell_z + k
                        inside = (0 <= x < shape[0]) and (
                            0 <= y < shape[1]) and (0 <= z < shape[2])

                        if inside:
                            value = grid.grid[x, y, z]
                            grid_ex[i + 2, j + 2, k + 2] = value
                            grid_dmi[i + 2, j + 2, k + 2] = value
                        else:
                            grid_ex[i + 2, j + 2, k + 2] = grid.grid[min(max(x, 0), shape[0]-1),
                                                                     min(max(
                                                                         y, 0), shape[1]-1),
                                                                     min(max(z, 0), shape[2]-1)]
                            grid_dmi[i + 2, j + 2, k + 2] = np.zeros(3)

        if grid.dmi_D is not None:
            # DMI constants (3x3x3) for the segmented 5x5x5 grid.
            D = grid.dmi_D[cell_x:cell_x+3, cell_y:cell_y+3, cell_z:cell_z+3]

        # 2. Original spin
        spins[0] = grid_ex[2, 2, 2]

        # 3. Proposal spin
        direction = grid_ex[2, 2, 2] + np.random.normal(0, 0.06, size=3)
        magnitude = np.sqrt(np.sum(direction**2))
        direction /= magnitude

        # direction = euler_rotate(grid_ex[2, 2, 2], 3.14)
        spins[1] = direction

        # 4. Change in energy
        delta_E = numpy_delta(grid, spins, grid_ex,
                              grid_dmi, D, zeeman_H)

        # 5. Decision
        if delta_E < 0:  # if energy is lower than previous energy, accept the change
            grid.grid[cell_x, cell_y, cell_z] = direction
        # if energy is higher than previous energy, accept the change with probability exp(-dE/kbT)
        else:
            if np.random.uniform(0, 1) < np.exp(-(delta_E)/(K_B*temperature)):
                grid.grid[cell_x, cell_y, cell_z] = direction
            else:  # revert the change
                grid.grid[cell_x, cell_y, cell_z] = spins[0]
    return grid.grid


@njit(fastmath=True)
def driver_numba(N: int, grid: np.ndarray, energy_func, zeeman_H: np.ndarray, anisotropy_K: np.float64,
                 anisotropy_u: np.ndarray, exchange_A: np.float64, dmi_D: np.ndarray, Dtype: str, Ms: np.float64,
                 dx: np.float64, dy: np.float64, dz: np.float64, temperature: np.float64) -> np.ndarray:
    """ Monte Carlo driver function for Numba implementation

    Args:
        N (int): Number of Monte Carlo steps
        grid (np.ndarray): 3D array of spins(vector field)
        energy_func (function): Energy function
        zeeman_H (np.ndarray): Zeeman field
        anisotropy_K (np.float64): Anisotropy constant
        anisotropy_u (np.ndarray): Anisotropy axis
        exchange_A (np.float64): Exchange constant
        dmi_D (np.ndarray): DMI constant
        Dtype (str): DMI crystal class
        Ms (np.float64): Saturation magnetisation
        dx (np.float64): Grid spacing in x direction
        dy (np.float64): Grid spacing in y direction
        dz (np.float64): Grid spacing in z direction
        temperature (np.float64): Temperature in Kelvin

    Returns:
        grid (np.ndarray): Relaxed system
    """
    spins = np.zeros(
        (2, 3), dtype='float64')  # array to store the spin and the proposed spin

    if dmi_D is None:
        D = None

    for _ in prange(N):
        # 1. Randomly select a cell
        cell_x = np.random.randint(0, grid.shape[0])
        cell_y = np.random.randint(0, grid.shape[1])
        cell_z = np.random.randint(0, grid.shape[2])

        # if the cell is empty, select another cell
        while np.all(grid[cell_x, cell_y, cell_z] == 0):
            cell_x = np.random.randint(0, grid.shape[0])
            cell_y = np.random.randint(0, grid.shape[1])
            cell_z = np.random.randint(0, grid.shape[2])

        # Segmenting the grid into 5x5x5 cells
        if 2 <= cell_x <= grid.shape[0] - 3 and 2 <= cell_y <= grid.shape[1] - 3 and 2 <= cell_z <= grid.shape[2] - 3:
            grid_ex = grid[cell_x-2:cell_x+3, cell_y -
                           2:cell_y+3, cell_z-2:cell_z+3].copy()
            grid_dmi = grid_ex

        else:
            # 5x5x5 grid for exchange with neumann boundary conditions
            grid_ex = np.empty((5, 5, 5, 3))
            # 5x5x5 grid for DMI with dirichlet boundary conditions
            grid_dmi = np.empty((5, 5, 5, 3))
            # this is done to avoid changing the indexes (cell_x, cell_y, cell_z) in the main grid
            for i in prange(-2, 3):
                for j in prange(-2, 3):
                    for k in prange(-2, 3):
                        x, y, z = cell_x + i, cell_y + j, cell_z + k
                        inside = 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2]

                        if inside:
                            value = grid[x, y, z]
                            grid_ex[i + 2, j + 2, k + 2] = value
                            grid_dmi[i + 2, j + 2, k + 2] = value
                        else:
                            grid_ex[i + 2, j + 2, k + 2] = grid[min(max(x, 0), grid.shape[0]-1),
                                                                min(max(y, 0),
                                                                    grid.shape[1]-1),
                                                                min(max(z, 0), grid.shape[2]-1)]
                            grid_dmi[i + 2, j + 2, k + 2] = np.zeros(3)
        # DMI constant for the segmented 5x5x5 grid
        if dmi_D is not None:
            D = dmi_D[cell_x:cell_x+3, cell_y:cell_y+3, cell_z:cell_z+3]

        # 2. Original spin
        spins[0] = grid_ex[2, 2, 2]
        # 3. Proposal spin
        direction = grid_ex[2, 2, 2] + np.random.normal(0, 0.06, size=3)
        magnitude = np.sqrt(np.sum(direction**2))
        direction = direction/magnitude
        # direction = euler_rotate(grid_ex[2, 2, 2], 1.6)
        spins[1] = direction

        # 4. Change in energy
        delta_E = energy_func(grid_ex, grid_dmi, spins, Ms, zeeman_H,
                              exchange_A, anisotropy_K, anisotropy_u, D, Dtype, dx, dy, dz)

        # 5. Decision
        if delta_E < 0:  # if energy is lower than previous energy, accept the change
            grid[cell_x, cell_y, cell_z] = direction
        # if energy is higher than previous energy, accept the change with probability exp(-dE/kbT)
        else:
            if np.random.uniform(0, 1) < np.exp(-(delta_E)/(K_B*temperature)):
                grid[cell_x, cell_y, cell_z] = direction
            else:  # revert the change
                grid[cell_x, cell_y, cell_z] = spins[0]

    return grid


@njit(fastmath=True)
def euler_rotate(vector, alpha):
    """
    Rotate a given vector using Euler angles to achieve uniform angular deviation.
    This optimized version combines the operations of the previous four functions.
    """

    # Generate zΔ and φΔ for uniform deviation
    theta = np.random.uniform(0, alpha)
    z_delta = np.cos(theta)
    phi_delta = np.random.uniform(0, 2*np.pi)

    # Compute the rotated vector [xΔ, yΔ, zΔ] directly
    sin_arccos_z_delta = np.sin(np.arccos(z_delta))
    x_delta = sin_arccos_z_delta * np.cos(phi_delta)
    y_delta = sin_arccos_z_delta * np.sin(phi_delta)

    # Compute the spherical coordinates of the input vector
    x, y, z = vector
    theta_i = np.arccos(z)
    phi_i = np.arctan2(y, x)

    # Compute the rotation matrices and combine them directly
    cos_phi_i = np.cos(phi_i)
    sin_phi_i = np.sin(phi_i)
    cos_theta_i = np.cos(theta_i)
    sin_theta_i = np.sin(theta_i)

    Rz_Ry = np.array([
        [cos_phi_i * cos_theta_i, -sin_phi_i, cos_phi_i * sin_theta_i],
        [sin_phi_i * cos_theta_i, cos_phi_i, sin_phi_i * sin_theta_i],
        [-sin_theta_i, 0, cos_theta_i]
    ])

    # Apply the combined rotation
    rotated_vector = np.dot(Rz_Ry, np.array([x_delta, y_delta, z_delta]))

    return rotated_vector
