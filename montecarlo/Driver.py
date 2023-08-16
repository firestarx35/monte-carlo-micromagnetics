
import numpy as np
from numba import njit, prange
# from tqdm import tqdm


#Global constants
Kb = 1.38064852e-23 #Boltzmann constant


@njit(fastmath=True)
def driver_numba(N, grid, energy_func, zeeman_K, anisotropic_K, anisotropic_u, exchange_A, dmi_D, Ms, dx, dy, dz, temperature):
    """ Monte Carlo driver function for Numba implementation

    Args: 
        N (int): Number of Monte Carlo steps
        grid (np.ndarray): 3D array of spins
        energy_func (function): Energy function to be used
        zeeman_K (np.ndarray): Zeeman constant
        anisotropic_K (np.ndarray): Anisotropy constant
        anisotropic_u (np.ndarray): Anisotropy axis
        exchange_A (np.ndarray): Exchange constant
        dmi_D (np.ndarray): Dzyaloshinskii-Moriya constant
        Ms (float): Saturation magnetisation
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        dz (float): Grid spacing in z direction
        temperature (float): Temperature in Kelvin
    
    Returns:
        grid (np.ndarray): Relaxed system
    """
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
    
            for i in prange(-2, 3):   #TODO: change back to prange
                for j in prange(-2, 3):
                    for k in prange(-2, 3):
                        x, y, z = cell_x + i, cell_y + j, cell_z + k
                        value = ( grid[x, y, z] if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2]
                                else grid[min(max(x, 0), grid.shape[0]-1),
                                        min(max(y, 0), grid.shape[1]-1),
                                        min(max(z, 0), grid.shape[2]-1)]
                                )
                        small_grid[i + 2, j + 2, k + 2] = value
        
        D = dmi_D[cell_x:cell_x+3, cell_y:cell_y+3, cell_z:cell_z+3]
        
        # 3. energy before the change
        spins[0] = small_grid[2, 2, 2]

        # 4. randomly select a direction
        direction = small_grid[2, 2, 2] + np.random.normal(0, 0.8, size=3)
        #normalise the vector
        magnitude = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        direction = direction/magnitude
        spins[1] = direction

        delta_E = energy_func(small_grid, spins, Ms, zeeman_K, exchange_A, anisotropic_K, anisotropic_u, D, dx, dy, dz)

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




@njit(fastmath=True)
def driver_numba2(N, grid, energy_func, zeeman_K, anisotropic_K, anisotropic_u, exchange_A, dmi_D, Ms, dx, dy, dz, temperature):
    """ Monte Carlo driver function for Numba implementation

    Args: 
        N (int): Number of Monte Carlo steps
        grid (np.ndarray): 3D array of spins
        energy_func (function): Energy function to be used
        zeeman_K (np.ndarray): Zeeman constant
        anisotropic_K (np.ndarray): Anisotropy constant
        anisotropic_u (np.ndarray): Anisotropy axis
        exchange_A (np.ndarray): Exchange constant
        dmi_D (np.ndarray): Dzyaloshinskii-Moriya constant
        Ms (float): Saturation magnetisation
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        dz (float): Grid spacing in z direction
        temperature (float): Temperature in Kelvin
    
    Returns:
        grid (np.ndarray): Relaxed system
    """
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

        if 2 <= cell_x <= grid.shape[0] - 3 and 2 <= cell_y <= grid.shape[1] - 3 and 2 <= cell_z <= grid.shape[2] - 3:
            grid_ex = grid[cell_x-2:cell_x+3, cell_y-2:cell_y+3, cell_z-2:cell_z+3]
            grid_dmi = grid_ex.copy()
        else:
            grid_ex = np.empty((5, 5, 5, 3))
            grid_dmi = np.empty((5, 5, 5, 3))
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        x, y, z = cell_x + i, cell_y + j, cell_z + k
                        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2]:
                            value = grid[x, y, z]
                        else:
                            value_ex = grid[min(max(x, 0), grid.shape[0]-1),
                                            min(max(y, 0), grid.shape[1]-1),
                                            min(max(z, 0), grid.shape[2]-1)]
                            value_dmi = np.zeros(3)
                            grid_ex[i + 2, j + 2, k + 2] = value if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2] else value_ex
                            grid_dmi[i + 2, j + 2, k + 2] = value if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2] else value_dmi

        D = dmi_D[cell_x:cell_x+3, cell_y:cell_y+3, cell_z:cell_z+3]
        
        # 3. energy before the change
        spins[0] = grid_ex[2, 2, 2]

        # 4. randomly select a direction
        direction = grid_ex[2, 2, 2] + np.random.normal(0, 0.8, size=3)
        #normalise the vector
        magnitude = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
        direction = direction/magnitude
        spins[1] = direction

        delta_E = energy_func(grid_ex, grid_dmi, spins, Ms, zeeman_K, exchange_A, anisotropic_K, anisotropic_u, D, dx, dy, dz)

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



@njit(fastmath=True)
def random_spin_3D_cone(v, alpha):
    """
    Rotate vector v in 3D by a random angle within [-alpha, alpha] forming a 3D cone.
    
    Parameters:
    - v: Original vector (numpy array of shape (3,))
    - alpha: Maximum rotation angle (in radians)
    
    Returns:
    - v_rot: Rotated vector (numpy array of shape (3,))
    """
    # Generate a random angle within [-alpha, alpha]
    delta_theta = np.random.uniform(-alpha, alpha)
    
    # Generate a random azimuthal angle in [0, 2*pi]
    delta_phi = np.random.uniform(0, 2 * np.pi)
    
    # Create a unit vector based on the random angles
    dx = np.sin(delta_theta) * np.cos(delta_phi)
    dy = np.sin(delta_theta) * np.sin(delta_phi)
    dz = np.cos(delta_theta)
    
    # Combine the unit vector with the original vector
    v_rot = v + np.array([dx, dy, dz])
    
    # Normalize the resulting vector
    v_rot = v_rot / (v_rot ** 2).sum() ** 0.5
    
    return v_rot
