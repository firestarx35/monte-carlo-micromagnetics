
import numpy as np
from numba import njit, prange
# from tqdm import tqdm
# from montecarlo.Grid import Grid
from montecarlo.Energies.numba_energies import zeeman_energy, anisotropic_energy, exchange_energy, dmi_energy
import discretisedfield as df

@njit(fastmath=True, parallel=True)
def normalise_vector(vector):
    magnitude = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    return vector/magnitude

@njit
def driver_numba(N, grid, zeeman_K, anisotropic_K, anisotropic_u, exchange_A, Dtype, dmi_D, m0, Ms, dx, dy, dz, Kb, temperature):
    
    E_before = zeeman_energy(grid, zeeman_K, m0, Ms, dx, dy, dz) + anisotropic_energy(grid, anisotropic_K, anisotropic_u, dx, dy, dz) + exchange_energy(grid, exchange_A, dx, dy, dz) + dmi_energy(grid, Dtype, dmi_D, dx, dy, dz)

    accepted = 0
    rejected = 0
    adaptive_width = 0.6
    
    for j in range(N):
       
        # 1. Randomly select a cell
        cell_x, cell_y, cell_z = np.random.randint(0, grid.shape[:3])

        while np.all(grid[cell_x , cell_y, cell_z] == 0): #if the cell is empty, select another cell
            cell_x, cell_y, cell_z = np.random.randint(0, grid.shape[:3])
        # 2.1 Randomly select a direction by introducing some uniform noise to the existing direction

        direction = grid[cell_x, cell_y, cell_z] + np.random.normal(-adaptive_width, adaptive_width, size=3)
        direction = normalise_vector(direction) #normalise the direction vector

        # 2.2. Change the direction of the cell
        prev_direction = np.copy(grid[cell_x, cell_y, cell_z])
        grid[cell_x, cell_y, cell_z] = direction
        
        # 3.3. Calculate the energy of the system after the change
        E_after = zeeman_energy(grid, zeeman_K, m0, Ms, dx, dy, dz) + anisotropic_energy(grid, anisotropic_K, anisotropic_u, dx, dy, dz) + exchange_energy(grid, exchange_A, dx, dy, dz) + dmi_energy(grid, Dtype, dmi_D, dx, dy, dz)
        # 4. If energy is lower than previous energy, accept the change
        delta_E = E_after - E_before

        if delta_E < 0: #if energy is lower than previous energy, accept the change
            E_before = E_after
            accepted += 1

        else: #if energy is higher than previous energy, accept the change with probability exp(-dE/kT)
            R = np.exp(-delta_E/(Kb*temperature))
            if np.random.uniform(0, 1) < R:
                E_before = E_after
                accepted += 1
            else:
                grid[cell_x, cell_y, cell_z] = prev_direction #revert the change
                rejected += 1
        

        accepted_ratio = accepted/(accepted+rejected)
        if accepted_ratio == 1:
            accepted_ratio = 0.99
        
        adaptive_width = adaptive_width*(0.5/(1-accepted_ratio))
    return grid, accepted_ratio