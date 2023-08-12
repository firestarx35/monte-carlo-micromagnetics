
import numpy as np
from numba import njit, prange
from tqdm import tqdm
# from montecarlo.Grid import Grid
# from montecarlo.Energies.numba_energies import 
from montecarlo.Energies.numba_energies import delta_energy#, zeeman_energy, anisotropic_energy, exchange_energy, dmi_energy, total_energy


#Global constants
Kb = 1.38064852e-23 #Boltzmann constant



# def driver_numpy(N, grid, zeeman_K, anisotropic_K, anisotropic_u, exchange_A, dmi_D, m0, Ms, dx, dy, dz, Kb, temperature):

#     E_before = delta_energy(small_grid, spins, Ms, zeeman_K, exchange_A, anisotropic_K, anisotropic_u, dmi_D, dx, dy, dz)

#     for i in tqdm(range(N)):
#         # 1. Randomly select a cell
#         cell_x, cell_y, cell_z = np.random.randint(0, grid.shape[:3]) 

#         while np.all(self.grid[cell_x , cell_y, cell_z] == 0): #if the cell is empty, select another cell
#             cell_x, cell_y, cell_z = np.random.randint(0, self.grid.shape[:3]) 
#         # 2Randomly select a direction by introducing some uniform noise to the existing direction

#         # direction = self.grid[cell_x, cell_y, cell_z] + np.random.normal(-0.1, 0.1, size=3)
#         direction = self.grid[cell_x, cell_y, cell_z] + np.random.uniform(-0.1, 0.1, size=3)

#         direction = direction/np.linalg.norm(direction) #normalise the direction vector

#         #3 Change the direction of the cell
#         prev_direction = np.copy(self.grid[cell_x, cell_y, cell_z])
#         self.grid[cell_x, cell_y, cell_z] = direction
        
#         # 4 Calculate the energy of the system after the change
#         E_after = self.total_energy()

#         # 5 If energy is lower than previous energy, accept the change
#         delta_E = E_after - E_before

#         if delta_E < 0: #if energy is lower than previous energy, accept the change
#             E_before = E_after

#         else: #if energy is higher than previous energy, accept the change with probability exp(-dE/kT)
#             R = np.exp(-delta_E/(Kb*self.temperature))
#             if np.random.uniform(0, 1) < R:
#                 E_before = E_after

#             else:
#                 self.grid[cell_x, cell_y, cell_z] = prev_direction #revert the change
        
#     return self.grid

@njit(fastmath=True)
def driver_numba(N, grid, zeeman_K, anisotropic_K, anisotropic_u, exchange_A, dmi_D, Ms, dx, dy, dz, temperature):
    spins = np.zeros((2, 3), dtype='float64')

    for _ in prange(N):
        # 1. Randomly select a cell
        cell_x = np.random.randint(0, grid.shape[0])
        cell_y = np.random.randint(0, grid.shape[1])
        cell_z = np.random.randint(0, grid.shape[2])

        while np.all(grid[cell_x , cell_y, cell_z] == 0): #if the cell is empty, select another cell
            cell_x = np.random.randint(0, grid.shape[0])
            cell_y = np.random.randint(0, grid.shape[1])
            cell_z = np.random.randint(0, grid.shape[2])

        # check if the cell is within 2 cells from the boundary
        if 2 <= cell_x <= grid.shape[0] - 3 and 2 <= cell_y <= grid.shape[1] - 3 and 2 <= cell_z <= grid.shape[2] - 3:
            small_grid = grid[cell_x-2:cell_x+3, cell_y-2:cell_y+3, cell_z-2:cell_z+3]
        else:   
            small_grid = np.empty((5, 5, 5, 3))
    
            for i in prange(-2, 3):
                for j in prange(-2, 3):
                    for k in prange(-2, 3):
                        x, y, z = cell_x + i, cell_y + j, cell_z + k
                        value = ( grid[x, y, z] if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2]
                                else grid[min(max(x, 0), grid.shape[0]-1),
                                        min(max(y, 0), grid.shape[1]-1),
                                        min(max(z, 0), grid.shape[2]-1)]
                                )
                        small_grid[i + 2, j + 2, k + 2] = value
        
        # 3. energy before the change
        spins[0] = small_grid[2, 2, 2]

        # 4. randomly select a direction
        direction = small_grid[2, 2, 2] + np.random.uniform(-0.1, 0.1, size=3)
        #normalise the vector
        magnitude = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        direction = direction/magnitude
        spins[1] = direction
        
        delta_E = delta_energy(small_grid, spins, Ms, zeeman_K, exchange_A, anisotropic_K, anisotropic_u, dmi_D, dx, dy, dz)

        # 6. Decision
        if delta_E < 0: #if energy is lower than previous energy, accept the change
            grid[cell_x, cell_y, cell_z] = direction #accept the change
        else: #if energy is higher than previous energy, accept the change with probability exp(-dE/kT)
            if np.random.uniform(0, 1) < np.exp(-(delta_E)/(Kb*temperature)):
                grid[cell_x, cell_y, cell_z] = direction #accept the change
            else:
                #reject the change
                grid[cell_x, cell_y, cell_z] = spins[0] #revert the change 
                
    return grid