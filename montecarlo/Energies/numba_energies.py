from numba import njit, prange
import numpy as np


#Global constants
m0 = 4*np.pi*1e-7 # Tm/A

@njit
def zeeman_energy(grid, zeeman_K, m0, Ms, dx, dy, dz):
    # create a grid shape validation for length of grids shape less than 2 and more than 4 are not supported
    energy = -m0 * Ms * np.sum(grid[1:-1, 1:-1, 1:-1]* zeeman_K) * dx * dy * dz
    return energy


@njit
def anisotropic_energy(grid, anisotropic_K, anisotropic_u, dx, dy, dz):

    energy = np.cross(grid[1:-1, 1:-1, 1:-1], anisotropic_u)
    energy = anisotropic_K*np.sum(energy**2)*dx*dy*dz #total energy of the system

    return energy


@njit
def exchange_energy(grid, exchange_A, dx, dy, dz):

    laplacian_M = (((grid[2:, 1:-1, 1:-1] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[:-2, 1:-1, 1:-1])/dx**2) + \
                    ((grid[1:-1, 2:, 1:-1] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[1:-1, :-2, 1:-1])/dy**2) + \
                    ((grid[1:-1, 1:-1, 2:] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[1:-1, 1:-1, :-2])/dz**2)
                )
                    
    energy = np.sum(grid[1:-1, 1:-1, 1:-1]*laplacian_M) #dot product of m and laplacian_M
    energy = -exchange_A*energy*dx*dy*dz #total energy of the system
    return energy


@njit
def dmi_Cnvz(grid, dmi_D, dx, dy, dz):

    gradM_z = np.empty_like(grid[1:-1,1:-1, 1:-1], dtype='float64')
    gradM_z[..., 0] = (grid[2:, 1:-1, 1:-1, 2] - grid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
    gradM_z[..., 1] = (grid[1:-1, 2:, 1:-1, 2] - grid[1:-1, :-2, 1:-1, 2]) / (2 * dy)
    gradM_z[..., 2] = (grid[1:-1, 1:-1, 2:, 2] - grid[1:-1, 1:-1, :-2, 2]) / (2 * dz)

    #divergence of m vector
    div_M = (
        (grid[2:, 1:-1, 1:-1, 0] - grid[:-2, 1:-1, 1:-1, 0]) / (2 * dx) +
        (grid[1:-1, 2:, 1:-1, 1] - grid[1:-1, :-2, 1:-1, 1]) / (2 * dy) +
        (grid[1:-1, 1:-1, 2:, 2] - grid[1:-1, 1:-1, :-2, 2]) / (2 * dz)
        )
    
    m_del_mz = np.sum(grid[1:-1, 1:-1, 1:-1]*gradM_z, axis=-1) #dot product of m and gradient of z component of m vector
    mz_div_m = grid[1:-1, 1:-1, 1:-1, 2]*div_M  # mz∇⋅m

    energy = dmi_D * (np.sum(m_del_mz) - np.sum(mz_div_m)) * dx * dy * dz #Total energy of the system
    
    return energy

@njit
def dmi_TO(grid, dmi_D, dx, dy, dz):

    curl = np.empty_like(self.grid, dtype='float64')
    curl[..., 0] = (grid[1:-1, 2:, 1:-1, 2] - grid[1:-1, :-2, 1:-1, 2]) / (2 * dy) - \
                            (grid[1:-1, 1:-1, 2:, 1] - grid[1:-1, 1:-1, :-2, 1]) / (2 * dz)
    
    curl[..., 1] = (grid[1:-1, 1:-1, 2:, 0] - grid[1:-1, 1:-1, :-2, 0]) / (2 * dz) - \
                            (grid[2:, 1:-1, 1:-1, 2] - grid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
    
    curl[..., 2] = (grid[2:, 1:-1, 1:-1, 1] - grid[:-2, 1:-1, 1:-1, 1]) / (2 * dx) - \
                            (grid[1:-1, 2:, 1:-1, 0] - grid[1:-1, :-2, 1:-1, 0]) / (2 * dy)
    
    energy= np.sum(grid*curl) # dot product of m and curl and summing over all the elements
    energy = dmi_D*energy*dx*dy*dz # total energy of the system


@njit
def delta_energy(grid, spins, Ms, zeeman_K, exchange_A, anisotropic_K, anisotropic_u, dmi_D, dx, dy, dz):
    """ Calculate the energy difference between the current state and the proposed state. Uses 'Conv_z' point group for DMI energy calculation.

    Args:
        grid (np.array): 3D vector field
        spins (np.array): (Previous and proposed) spin configuration
        m0 (float): Magnitude of the spin
        Ms (float): Saturation magnetisation
        zeeman_K (np.array): Zeeman constant
        exchange_A (float): Exchange constant
        anisotropic_K (float): Anisotropy constant
        anisotropic_u (np.array): Anisotropy axis
        dmi_D (float): DMI constant
        dx (float): x dimension of the grid
        dy (float): y dimension of the grid
        dz (float): z dimension of the grid

    Returns:
        float: Energy difference between the current state and the proposed state
    """

    delta = np.array([0,0], dtype='float64')

    for i in prange(spins.shape[0]):
        grid[2, 2, 2] = spins[i]
 
        zeeman = -m0 * Ms * np.sum(grid[1:-1, 1:-1, 1:-1]* zeeman_K) * dx * dy * dz
        
        aniso = np.cross(grid[1:-1, 1:-1, 1:-1], anisotropic_u)
        aniso = anisotropic_K*np.sum(aniso**2)*dx*dy*dz
        
        laplacian_M = (((grid[2:, 1:-1, 1:-1] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[:-2, 1:-1, 1:-1])/dx**2) + \
                        ((grid[1:-1, 2:, 1:-1] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[1:-1, :-2, 1:-1])/dy**2) + \
                        ((grid[1:-1, 1:-1, 2:] - 2 * grid[1:-1, 1:-1, 1:-1] + grid[1:-1, 1:-1, :-2])/dz**2)
                    )
        #dot product of m and laplacian_M               
        exchange = np.sum(grid[1:-1, 1:-1, 1:-1]*laplacian_M) 
        exchange = -exchange_A*exchange*dx*dy*dz #total energy of the system
        
        gradM_z = np.empty_like(grid[1:-1,1:-1, 1:-1], dtype='float64')
        gradM_z[..., 0] = (grid[2:, 1:-1, 1:-1, 2] - grid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
        gradM_z[..., 1] = (grid[1:-1, 2:, 1:-1, 2] - grid[1:-1, :-2, 1:-1, 2]) / (2 * dy)
        gradM_z[..., 2] = (grid[1:-1, 1:-1, 2:, 2] - grid[1:-1, 1:-1, :-2, 2]) / (2 * dz)

        #divergence of m vector
        div_M = ((grid[2:, 1:-1, 1:-1, 0] - grid[:-2, 1:-1, 1:-1, 0]) / (2 * dx) +
                (grid[1:-1, 2:, 1:-1, 1] - grid[1:-1, :-2, 1:-1, 1]) / (2 * dy) +
                (grid[1:-1, 1:-1, 2:, 2] - grid[1:-1, 1:-1, :-2, 2]) / (2 * dz)
                )
        #dot product of m and gradient of z component of m vector
        m_del_mz = np.sum(grid[1:-1, 1:-1, 1:-1]*gradM_z) 
        mz_div_m = np.sum(grid[1:-1, 1:-1, 1:-1, 2]*div_M)  # mz∇⋅m

        dmi = dmi_D * (m_del_mz - mz_div_m) * dx * dy * dz #Total energy of the system

        delta[i] = zeeman + aniso + exchange + dmi
    
    return delta[1] - delta[0]

@njit
def delta_energy2(grid, spins, Ms, zeeman_K, exchange_A, anisotropic_K, anisotropic_u, dmi_D, dx, dy, dz):
    """ Calculate the energy difference between the current state and the proposed state. Uses 'T or O' point group for DMI energy calculation.
        For Juan et al. 2022
    Args:
        grid (np.array): 3D vector field
        spins (np.array): (Previous and proposed) spin configuration
        m0 (float): Magnitude of the spin
        Ms (float): Saturation magnetisation
        zeeman_K (np.array): Zeeman constant
        exchange_A (float): Exchange constant
        anisotropic_K (float): Anisotropy constant
        anisotropic_u (np.array): Anisotropy axis
        dmi_D (float): DMI constant
        dx (float): x dimension of the grid
        dy (float): y dimension of the grid
        dz (float): z dimension of the grid

    Returns:
        float: Energy difference between the current state and the proposed state
    """

    delta = np.array([0,0], dtype='float64')

    for i in prange(spins.shape[0]):
        grid[2, 2, 2] = spins[i]
 
        zeeman = -m0 * Ms * np.sum(grid[1:-1, 1:-1, 1:-1]* zeeman_K) * dx * dy * dz

        gradM = ((grid[2:, 1:-1, 1:-1] - grid[:-2, 1:-1, 1:-1]) / (2 * dx) +
                (grid[1:-1, 2:, 1:-1] - grid[1:-1, :-2, 1:-1]) / (2 * dy) +
                (grid[1:-1, 1:-1, 2:] - grid[1:-1, 1:-1, :-2]) / (2 * dz)
            )          
        exchange = np.sum(gradM**2)
        exchange = -exchange_A*exchange*dx*dy*dz #total energy of the system
        
        curl = np.empty_like(grid[1:-1, 1:-1, 1:-1], dtype='float64')
        curl[..., 0] = (grid[1:-1, 2:, 1:-1, 2] - grid[1:-1, :-2, 1:-1, 2]) / (2 * dy) - \
                                (grid[1:-1, 1:-1, 2:, 1] - grid[1:-1, 1:-1, :-2, 1]) / (2 * dz)
        
        curl[..., 1] = (grid[1:-1, 1:-1, 2:, 0] - grid[1:-1, 1:-1, :-2, 0]) / (2 * dz) - \
                                (grid[2:, 1:-1, 1:-1, 2] - grid[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
        
        curl[..., 2] = (grid[2:, 1:-1, 1:-1, 1] - grid[:-2, 1:-1, 1:-1, 1]) / (2 * dx) - \
                                (grid[1:-1, 2:, 1:-1, 0] - grid[1:-1, :-2, 1:-1, 0]) / (2 * dy)
        
        dmi = np.sum(grid*curl) # dot product of m and curl and summing over all the elements
       
        dmi = dmi_D*dmi*dx*dy*dz # total energy of the system

        delta[i] = zeeman + exchange + dmi
    
    return delta[1] - delta[0]