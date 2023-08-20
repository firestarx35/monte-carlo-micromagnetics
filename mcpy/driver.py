
import numpy as np
from numba import njit, prange
from tqdm import tqdm
from mcpy.energies.numpy_energies import numpy_delta


"""This module contains the main functions for implementing the Metropolis-Hastings Monte Carlo simulations. 
By changing the spin each iteartion by random angle with uniform distribution."""

# Global constants

K_B = 1.38064852e-23  # Boltzmann constant


def driver_numpy(N, grid, zeeman_H, temperature):
    """ Monte Carlo driver function for Numpy implementation

    Args: 
        N (int): Number of Monte Carlo steps
        grid (Grid): mcpy.Grid object
        zeeman_H (np.ndarray): Zeeman field
        temperature (float): Temperature in Kelvin

    Returns:
        grid (np.ndarray): Relaxed system
    """
    spins = np.zeros(
        (2, 3), dtype='float64')  # array to store original and proposed spin
    shape = grid.grid.shape

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

        # DMI constants (3x3x3) for the segmented 5x5x5 grid.
        D = grid.dmi_D[cell_x:cell_x+3, cell_y:cell_y+3, cell_z:cell_z+3]

        # 2. Original spin
        spins[0] = grid_ex[2, 2, 2]

        # 3. Proposal spin
        direction = grid_ex[2, 2, 2] + np.random.uniform(-0.2, 0.2, size=3)
        magnitude = np.sqrt(np.sum(direction**2))
        direction /= magnitude
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
def driver_numba(N, grid, energy_func, zeeman_H, anisotropy_K, anisotropy_u, exchange_A, dmi_D, Dtype, Ms, dx, dy, dz, temperature):
    """ Monte Carlo driver function for Numba implementation

    Args: 
        N (int): Number of Monte Carlo steps
        grid (np.ndarray): 3D array of spins
        energy_func (function): Energy function to be used
        zeeman_H (np.ndarray): Zeeman field
        anisotropy_K (np.ndarray): Anisotropy constant
        anisotropy_u (np.ndarray): Anisotropy axis
        exchange_A (np.ndarray): Exchange constant
        dmi_D (np.ndarray): Dzyaloshinskii-Moriya constant
        Dtype (np.dtype): DMI type or Crystal class
        Ms (float): Saturation magnetisation
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        dz (float): Grid spacing in z direction
        temperature (float): Temperature in Kelvin

    Returns:
        grid (np.ndarray): Relaxed system
    """
    spins = np.zeros(
        (2, 3), dtype='float64')  # array to store the spin and the proposed spin

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
        D = dmi_D[cell_x:cell_x+3, cell_y:cell_y+3, cell_z:cell_z+3]

        # 2. Original spin
        spins[0] = grid_ex[2, 2, 2]
        # 3. Proposal spin
        direction = grid_ex[2, 2, 2] + np.random.uniform(-0.2, 0.2, size=3)
        magnitude = np.sqrt(np.sum(direction**2))
        direction = direction/magnitude
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
def random_spin_uniform(v, alpha):
    # Sample the cosine of the polar angle uniformly
    cos_del0 = np.random.uniform(np.cos(alpha), 1)
    del_phi = np.random.uniform(0, 2 * np.pi)

    # Derive the polar angle from its cosine value
    del0 = np.arccos(cos_del0)
    dx = np.sin(del0) * np.cos(del_phi)
    dy = np.sin(del0) * np.sin(del_phi)
    dz = np.cos(del0)

    # Combine the unit vector with the original vector
    v_proposal = v + np.array([dx, dy, dz])

    # Normalising
    v_proposal = v_proposal / np.sqrt(np.sum(v_proposal ** 2))

    return v_proposal
